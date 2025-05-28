import sys, os, json
from openai import OpenAI
import time
import pandas as pd
from pprint import pprint
import importlib
from sentence_transformers import SentenceTransformer
prsp_gen = importlib.import_module("gpt-prsp-gen-ContextQ")
suitability_checker = importlib.import_module("gpt-q-suitability-checker")

class StoryPlotSegmentation:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4.5-preview-2025-02-27"
        self.temperature = 0.7
        # self.max_tokens = 1000
        self.top_p = .8
        # self.frequency_penalty = 0
        # self.presence_penalty = 0
        self.moral_prompt = self._load_prompt("./prompts/moralQ_prompts/story_moral_prompt.txt")
        self.q_prompt = self._load_prompt("./prompts/moralQ_prompts/story_moral_q_prompt.txt")
        self.checker = suitability_checker.GPTPedagogicQSuitabilityChecker()

    def _load_prompt(self, prompt_path):
        with open(prompt_path, "r") as f:
            return f.read()
    
    def _load_story_content(self, story_path):
        pages = sorted(list(filter(lambda x: x.endswith(".json"), os.listdir(story_path))))
        story_content = []
        q_by_page = dict()
        for i in range(1, len(pages)):
            with open(os.path.join(story_path, pages[i]), "r") as f:
                story_page_content = json.load(f)
            story_content.append(' '.join(story_page_content["text"]))
            q_by_page[i] = []
            for prompt in story_page_content["prompts"]:
                if "question_type" in prompt.keys() and prompt["question_type"] == "open-ended":
                    if "is_checkpoint_question" not in prompt.keys() or prompt["is_checkpoint_question"] != True:
                        q_by_page[i].append(prompt["question"])
        return story_content, q_by_page

    def _load_all_page_contextQ_questions(self, contextQ_path):
        if not os.path.exists(contextQ_path):
            return []
        
        with open(contextQ_path, "r") as f:
            contextQ = json.load(f)
        page_questions = [q["prompt"] for page in contextQ.keys() for q in contextQ[page]]
        return page_questions
    
    def _increase_randomness_llm_param(self):
        self.temperature = .9
    
    def _reset_llm_param(self):
        self.temperature = .7

    def generate_story_moral(self, story, segment_heuistic):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.moral_prompt.replace("$STORY$", '\n'.join(story)).replace("$SEGMENT_HEURISTIC$", segment_heuistic)}],
            temperature=self.temperature,
            # max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        return response.choices[0].message.content
    
    def generate_moral_q(self, previous_events, current_event, regeneration_str=""):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.q_prompt.replace("$PREVIOUS_PAGES$", previous_events).replace("$CURRENT_TEXT$", current_event).replace("$REGENERATION_PROMPT$", regeneration_str)}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content    

    def generate_story_segment_questions(self, previous_events, current_event, existing_questions=[], num_questions=1):
        generated_questions = []
        # Maximum tries
        maximum_tries = 5
        failed = 0
        regeneration_explanation = []
        self._reset_llm_param()
        while len(generated_questions) < num_questions:
            if failed >= maximum_tries:
                break
            elif failed > 0:
                print("......Regenerating... for the ", failed+1, " time")
                if failed >= 2:
                    self._increase_randomness_llm_param()
            
            try:
                response = json.loads(self.generate_moral_q(previous_events, current_event, "\n".join(regeneration_explanation) + " Please avoid the issues in the previous bad attempts."))
                candidate_prompt = response["prompt"]
            except Exception as e:
                print("Error generating a QnA for segment")
                failed += 1
                continue
            

            # First check if the question is repeated
            is_unique, explanation = self.checker.check_uniqueness(current_event, candidate_prompt, existing_questions, generated_questions)
            if is_unique == False:
                failed += 1
                regeneration_explanation.append(explanation)
                continue
            
            # Then check if the question is suitable
            passed, results = self.checker.check_all_suitability(previous_events, current_event, candidate_prompt)
            if passed:
                generated_questions.append(candidate_prompt)
            else:
                failed += 1
                regeneration_explanation.append(self.checker.get_regeneration_prompt(results))
                continue
            
            if len(generated_questions) >= num_questions:
                break

        return generated_questions
    
def generate_moral_and_segment_story(segmentor, story_content, output_path):
    def _quant_segment_criticizer(segments):
        pages = []
        last_start = 0
        for _id in segments["segments"].keys():
            segment = segments["segments"][_id]
            if segment["START"] <= last_start:
                return False
            pages += list(range(segment["START"], segment["END"] + 1))
            last_start = segment["END"]
        if len(pages) != len(story_content):
            print("......Segmentation is not correct for story: ", output_path.split("/")[-1])
            print("......Pages: ", pages)
            print("......Story content: ", len(story_content))
            return False
        return True

    def _qual_segment_criticizer(segments):
        print("......Moral of the story: ", segments["moral"])
        for segment in segments["segments"].keys():
            print("......Segment: ", segment)
            print("......START, END: ", segments["segments"][segment]["START"], segments["segments"][segment]["END"])
            print("......SUMMARY: ", segments["segments"][segment]["SUMMARY"])
            print("......REASONING: ", segments["segments"][segment]["REASONING"])
        feedback = input("......Enter your feedback: ")
        if feedback in ["y", "Y", "yes", "Yes", "YES"]:
            return None
        else:
            return feedback

    story_name = output_path.split("/")[-1].split(".")[0]
    if os.path.exists(output_path):
        print("......Segments file already exists for story: ", story_name)
        with open(output_path, "r") as f:
            segments = json.load(f)
        return segments
    
    numbered_content = [str(i) + ". " + content for i, content in enumerate(story_content)]
    segment_heuistic = ""
    # Step 1: Generate the moral for the story
    for i in range(3):
        response = segmentor.generate_story_moral(numbered_content, segment_heuistic)
        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "")
        try:
            segments = json.loads(response)
        except:
            print("Error parsing segments for story: ", story_name)
            print(response)
            print("......Re-generating moral for story: ", story_name)
        
        # Criticizer
        if segments is not None:
            if not _quant_segment_criticizer(segments):
                segments = None
                print("......Regenerating because the segmentation is not correct for story: ", story_name)
                continue
            else:
                feedback = _qual_segment_criticizer(segments)
                if feedback is None:
                    # No feedback means the segmentation is good
                    break
                else:
                    segment_heuistic = feedback
                    segments = None
                    continue

    # Step 2: Segment out the story content for each segment
    for segment_id in segments["segments"].keys():
        segment = segments["segments"][segment_id]
        segments["segments"][segment_id]["TEXT"] = story_content[segment["START"]-1:segment["END"]]

    if segments is not None:
        with open(output_path, "w") as f:
            json.dump(segments, f, indent=4)
    else:
        print("Error generating segments for story: ", story_name)
    return segments

def generate_moral_q_for_story(segmentor, story_segments, moral_q_output, page_questions):
    prior_segment_questions = []
    if os.path.exists(moral_q_output):
        with open(moral_q_output, "r") as f:
            est_moral_q = json.load(f)
        processed = True
        for segment_id in est_moral_q["segments"].keys():
            if "questions" not in est_moral_q["segments"][segment_id].keys():
                processed = False
                break
            else:
                prior_segment_questions += est_moral_q["segments"][segment_id]["questions"]
        if processed:
            print("......Moral Q already generated for story: ", moral_q_output)
            return est_moral_q
        else:
            story_segments = est_moral_q
    
    print("......Generating Moral Q for story: ", moral_q_output.split("/")[-1])
    previous_events = []
    segment_ids = sorted(list(story_segments["segments"].keys()))
    NUMBER_OF_QUESTIONS = 2
    for segment_id in segment_ids:
        segment = story_segments["segments"][segment_id]
        existing_questions = prior_segment_questions + page_questions
        # Check if the questions are already generated
        if "questions" in segment.keys() and len(segment["questions"]) >= NUMBER_OF_QUESTIONS:
            continue

        print("......Generating Moral Q for segment: ", segment_id)
        segment_questions = segmentor.generate_story_segment_questions('\n'.join(previous_events), segment["SUMMARY"], existing_questions, num_questions=NUMBER_OF_QUESTIONS)
        if len(segment_questions) < NUMBER_OF_QUESTIONS:
            print("**********"*5)
            print("......Not enough questions generated for segment: ", segment_id, " with ", len(segment_questions), " questions")
        story_segments["segments"][segment_id]["questions"] = segment_questions
        previous_events.append(segment["SUMMARY"])
        prior_segment_questions += segment_questions

        with open(moral_q_output, "w") as f:
            json.dump(story_segments, f, indent=4)
    
    return story_segments

def aggregate_story_segments(story_segments, story_name):
    """
    Sometimes the story is segmented into too many parts.
    """
    def _combine_several_segments(segments, start_id, end_id):
        combined_segment = {
            "START": segments["segments"]["segment_" + str(start_id)]["START"],
            "END": segments["segments"]["segment_" + str(end_id)]["END"],
            "REASONING": "",
            "SUMMARY": "",
            "TEXT": [],
            "questions": []
        }

        selected_questions = []
        for i in range(start_id, end_id + 1):
            combined_segment["REASONING"] += segments["segments"]["segment_" + str(i)]["REASONING"]
            combined_segment["SUMMARY"] += segments["segments"]["segment_" + str(i)]["SUMMARY"]
            combined_segment["TEXT"] += segments["segments"]["segment_" + str(i)]["TEXT"]
            combined_segment["questions"] += segments["segments"]["segment_" + str(i)]["questions"]
            if "Q_SELECTED_BY_xiajie" in segments["segments"]["segment_" + str(i)].keys():
                selected_questions.append(segments["segments"]["segment_" + str(i)]["Q_SELECTED_BY_xiajie"])
        
        # Select the questions from the selected questions
        if len(selected_questions) > 0:
            print("Arbitrate question for combined segment: ", combined_segment["SUMMARY"])
            for i in range(len(selected_questions)):
                print(str(i) + ". " + selected_questions[i]["question"])
            question_index = int(input("Enter the index of the question you want to select (numeric number, -1 or outbound for none): "))
            if question_index < len(selected_questions) and question_index >= 0:
                combined_segment["Q_SELECTED_BY_xiajie"] = {
                    "index": -1,
                    "question": selected_questions[question_index]["question"]
                }
            else:
                print("Invalid question index: ", question_index)
                rewrite = input("Please enter the question you want to write for this segment: ")
                if rewrite != "":
                    combined_segment["Q_SELECTED_BY_xiajie"] = {
                        "index": -1,
                        "question": rewrite
                    }
        
        return combined_segment
    
    segment_length = [segment["END"] - segment["START"] + 1 for segment in story_segments["segments"].values()]
    if len(segment_length) >= 4 or min(segment_length) <= 2:
        print("......Might need to aggregate segments for story: ", story_name)
        aggregated_segments = dict()
        current_segments = list(story_segments["segments"].keys())
        aggregation_list = []
        while True:
            if input("Do you want to add segments for aggregation? (y/n): ") in ["y", "Y", "yes", "Yes", "YES"]:
                start_segment = input("Enter the start segment id for aggregation: ")
                end_segment = input("Enter the end segment id for aggregation:")
                assert end_segment > start_segment, "End segment id must be greater than start segment id"
                aggregation_list.append((int(start_segment), int(end_segment)))
            else:
                break
        if len(aggregation_list) == 0:
            print("...No need to aggregate segments for this story")
            return False, story_segments
        
        segment_list = []
        aggregating_ids = aggregation_list.pop(0)
        print(aggregating_ids)
        for i in range(1, len(current_segments) + 1):
            if i < aggregating_ids[0]:
                segment_list.append(story_segments["segments"][current_segments[i-1]])
            elif i == aggregating_ids[0]:
                segment_list.append(_combine_several_segments(story_segments, aggregating_ids[0], aggregating_ids[1]))
            elif i > aggregating_ids[0] and i < aggregating_ids[1]:
                continue
            elif i == aggregating_ids[1]:
                if len(aggregation_list) > 0:
                    aggregating_ids = aggregation_list.pop(0)
                    continue
            else:
                segment_list.append(story_segments["segments"][current_segments[i-1]])
        print(segment_list)

        for i in range(len(segment_list)):
            aggregated_segments["segment_" + str(i+1)] = segment_list[i]
        
        story_segments["segments"] = aggregated_segments
        return True, story_segments
    else:
        print("...No need to aggregate segments for this story")
        return False, story_segments
    

def arbitrate_questions(story_segments, moral_q_output, arbitrator="xiajie"):
    for segment_id in story_segments["segments"].keys():
        segment = story_segments["segments"][segment_id]
        if "Q_SELECTED_BY_" + arbitrator in segment.keys():
            print("......Question already selected by ", arbitrator)
            continue

        print("......Selecting a question from QnA for segment: ", segment_id)
        print("......Summary: ", segment["SUMMARY"])
        for i in range(len(segment["questions"])):
            print(str(i) + ". " + segment["questions"][i])
        question_index = int(input("Enter the index of the question you want to select (numeric number, -1 or outbound for none): "))
        if question_index < len(segment["questions"]) and question_index >= 0:
            story_segments["segments"][segment_id]["Q_SELECTED_BY_" + arbitrator] = {
                "index": question_index,
                "question": segment["questions"][question_index],
            }
        else:
            print("Invalid question index: ", question_index)
            rewrite = input("Do you want to rewrite the question? (y/n): ")
            rewrite_question = None
            if rewrite in ["y", "Y", "yes", "Yes", "YES"]:
                rewrite_question = input("Enter the new question: ")
            story_segments["segments"][segment_id]["Q_SELECTED_BY_" + arbitrator] = {
                "index": -1,
                "question": rewrite_question
            }
        with open(moral_q_output, "w") as f:
            json.dump(story_segments, f, indent=4)
    return story_segments

def generate_perspectives(story_segments, moral_q_prsp_output):
    prsp_generator = prsp_gen.PerspectivesGeneration()
    for segment_id in story_segments.keys():
        segment = story_segments[segment_id]
        print("......Generating perspectives for segment: ", segment_id)
        for i in range(len(segment["questions"])):
            perspectives = prsp_generator.generate_perspectives(segment["questions"][i]["prompt"], segment["SUMMARY"])
            story_segments[segment_id]["questions"][i]["perspectives"] = perspectives

    with open(moral_q_prsp_output, "w") as f:
        json.dump(story_segments, f, indent=4)
    return story_segments


def main():
    asset_path = "/Users/xiajie/Projects/github-repos/interactive-storybook-assets/qna_json/"
    contextQ_path = "./outputs_q/rep_checked_ContextQ"
    output_path = "./outputs_q_moral"
    segment_output_path = os.path.join(output_path, "story_segments_by_moral_2")
    moral_q_output_path = os.path.join(output_path, "story_moral_q_2")
    if not os.path.exists(segment_output_path):
        os.makedirs(segment_output_path)
    if not os.path.exists(moral_q_output_path):
        os.makedirs(moral_q_output_path)
    segmentor = StoryPlotSegmentation()
    educare_storybooks = ["ada_twist_scientist", "a_letter_to_amy", "boxitects", "grumpy_monkey", "if_you_give_a_mouse_a_cookie", "jabari_jumps", "last_stop_on_market_street", "mango_abuela_and_me", "peters_chair", "stand_tall_molly_lou_melon", "the_proudest_blue"]

    kipp_storybooks = ["chicka_chicka_boom_boom", "the_little_red_hen-muldrow", "rap_a_tap_tap_heres_bojangles",
                       "three_little_pigs-marshall", "the_three_billy_goats_gruff-finch", "ganeshas_sweet_tooth",
                       "moon_rope", "the_story_of_ferdinand", "helpers_in_my_community", "a_day_in_the_life_of_a_firefighter",
                      "my_five_senses-aliki"]
    stories = kipp_storybooks + educare_storybooks

    arbitrate = True
    arbitrated_by = ""
    for story in stories:
        print("\n...Processing story: ", story)
        story_input_path = os.path.join(asset_path, story)
        story_segment_output = os.path.join(segment_output_path, story + ".json")
        story_content, _ = segmentor._load_story_content(story_input_path)
        page_questions = segmentor._load_all_page_contextQ_questions(os.path.join(contextQ_path, story + ".json"))
        print("......Story has ", len(page_questions), " page questions")

        # STEP 1: Generating Segments: Skip if the segments file already exists
        story_segments = generate_moral_and_segment_story(segmentor, story_content, story_segment_output)

        # STEP 2: Generating QnA: Skip if the QnA file already exists
        story_moral_q_output = os.path.join(moral_q_output_path, story + ".json")
        story_moral_q = generate_moral_q_for_story(segmentor, story_segments, story_moral_q_output, page_questions)

        # # STEP : Aggregating segments - some segments are too short for questions
        # #       Skip if the aggregated segments file already exists
        # aggregated, new_aggregated_segments = aggregate_story_segments(story_moral_q, story)
        # if aggregated:
        #     with open(story_moral_q_output, "w") as f:
        #         json.dump(new_aggregated_segments, f, indent=4)
        #     story_moral_q = new_aggregated_segments

        if arbitrate:
            # Step 3-1: If multiple questions are generated, arbitrate for the best question
            if arbitrated_by == "":
                arbitrated_by = input("Enter your name: ").replace(" ", "_").lower()
            story_moral_q = arbitrate_questions(story_moral_q, story_moral_q_output, arbitrated_by)

        # STEP 3: Generate Perspectives for each moral question
        # story_moral_q_prsp_output = os.path.join(moral_q_prsp_output_path, story + ".json")
        # story_moral_q_prsp = generate_perspectives(story_moral_q, story_moral_q_prsp_output)

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: python gpt-q-gen-MoralQ.py")
        sys.exit(1)
    main()

