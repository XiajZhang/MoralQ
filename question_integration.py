import sys, os
import json
import importlib
suitability_checker = importlib.import_module("gpt-q-suitability-checker")

def _mockup_story_qna_item(prompt, current_idx):
    template = {
            "type": 5,
            "id": -1,
            "hint": "",
            "sentence": [],
            "keyword": [],
            "highlight": {
                "textId": [],
                "sentenceId": [],
                "sceneObjectId": []
            },
            "question": "",
            "child_response": {
                "textId": [],
                "sceneObjectId": [],
                "speech": []
            },
            "robot_response": {
                "textId": [],
                "sceneObjectId": [],
                "speech": ""
            },
            "is_checkpoint_question": False,
            "question_type": "open-ended"
        }
    template["question"] = prompt
    template["id"] = current_idx
    return template

def _load_original_storybook_json(storybook_path):
    """
    Load the original storybook JSON
    """
    with open(storybook_path, "r") as f:
        return json.load(f)

# def integrate_perspectives_into_storybook(storybook_qna_path):
#     """
#     Integrate the perspectives into the storybook QnA
#     """

#     generated_prompts_folder = "./perspectives"
#     generated_prompts = list(filter(lambda x: x.endswith("prompt.json"), os.listdir(generated_prompts_folder)))
#     storybooks = list(map(lambda x: x.replace("_prompt.json", ""), generated_prompts))
#     print(storybooks)

#     for story in storybooks:
#         print("\nProcessing story " + story)
#         story_qnas = sorted(list(os.listdir(os.path.join(storybook_qna_path, story))))

#         # Newly generated prompts
#         with open(os.path.join(generated_prompts_folder, story + "_prompt.json"), "r") as f:
#             llm_prompt_data = json.load(f)
        
        
#         for qna_file in story_qnas[1:]:
#             with open(os.path.join(storybook_qna_path, story, qna_file), "r") as f:
#                 storybook = json.load(f)


#             new_prompts_list = list(filter(lambda x: x["type"] != 5 or x["question_type"] != "open-ended", storybook["prompts"]))
            
#             if qna_file in llm_prompt_data.keys():
#                 new_prompt = _mockup_story_qna_item(llm_prompt_data[qna_file], len(new_prompts_list))
#                 new_prompts_list.append(new_prompt)

#             storybook["prompts"] = new_prompts_list

#             with open(os.path.join(storybook_qna_path, story, qna_file), "w") as f:
#                 json.dump(storybook, f, indent=4)
        
def integrate_page_questions_into_storybook(new_question_path, storybook_json_dir):
    """
    Integrate the page questions into the storybook QnA
    """
    def _get_and_clean_up_prior_questions(storybook_json):
        prompts = []
        for prompt in storybook_json["prompts"]:
            if prompt["type"] == 5 and prompt["question_type"] == "open-ended":
                continue
            replica_prompt = prompt.copy()
            replica_prompt["is_checkpoint_question"] = False
            replica_prompt["question_type"] = "scripted"
            prompts.append(replica_prompt)
        return prompts

    with open(new_question_path, "r") as f:
        new_questions = json.load(f)

    print("...Processing page question integration for story " + new_question_path.split("/")[-1])
    for story_page in new_questions.keys():
        storybook_json_path = os.path.join(storybook_json_dir, story_page)
        with open(storybook_json_path, "r") as f:
            storybook_json = json.load(f)
        prior_prompts = _get_and_clean_up_prior_questions(storybook_json)
        for question in new_questions[story_page]:
            new_prompt = _mockup_story_qna_item(question["prompt"], len(prior_prompts))
            prior_prompts.append(new_prompt)
        
        storybook_json["prompts"] = prior_prompts
        with open(storybook_json_path, "w") as f:
            json.dump(storybook_json, f, indent=4)

def integrate_moral_questions_into_storybook_json(story_name, moral_questions_dir, qna_json_dir):
    with open(os.path.join(moral_questions_dir, story_name + ".json"), "r") as f:
        moral_q_json = json.load(f)
    page_jsons = list(filter(lambda x: x.endswith(".json"), os.listdir(os.path.join(qna_json_dir, story_name))))

    # First check story page count and moral segments
    print("...Processing moral question integration for story " + story_name)
    pages = []
    moral_questions = dict()
    for segment_id in moral_q_json["segments"].keys():
        segment = moral_q_json["segments"][segment_id]
        pages += list(range(segment["START"], segment["END"] + 1))
        moral_question_page = story_name + "_" + str(segment["END"]) + ".json" if segment["END"] > 9 else story_name + "_0" + str(segment["END"]) + ".json"
        assert moral_question_page in page_jsons, "Story QnA file not found for story " + moral_question_page
        moral_questions[moral_question_page] = segment["Q_SELECTED_BY_xiajie"]["question"]
    assert len(pages) + 1 == len(page_jsons), "Page count mismatch between storybook and moral questions: " + str(len(pages)) + " != " + str(len(page_jsons))

    for page in moral_questions.keys():
        with open(os.path.join(qna_json_dir, story_name, page), "r") as f:
            story_json = json.load(f)
        segment_question = _mockup_story_qna_item(moral_questions[page], len(story_json["prompts"]))
        segment_question["is_checkpoint_question"] = True
        story_json["prompts"].append(segment_question)

        with open(os.path.join(qna_json_dir, story_name, page), "w") as f:
            json.dump(story_json, f, indent=4)

def check_duality_of_storybook_json_open_ended_questions(qna_json_dir):
    def _remove_duplicate_prompts(story_json_dir, story_duplicate_prompts):
        if len(story_duplicate_prompts.keys()) == 0:
            return
        
        new_prompts = []
        for page in story_duplicate_prompts.keys():
            with open(os.path.join(story_json_dir, page), "r") as f:
                story_json = json.load(f)
            removing_prompts = [prompt["prompt"] for prompt in story_duplicate_prompts[page]]
            for prompt in story_json["prompts"]:
                if prompt["question"] not in removing_prompts or prompt["is_checkpoint_question"]:
                    _new_prompt = prompt.copy()
                    _new_prompt["id"] = len(new_prompts)
                    new_prompts.append(_new_prompt)
        story_json["prompts"] = new_prompts
        with open(os.path.join(story_json_dir, page), "w") as f:
            json.dump(story_json, f, indent=4)

    """
    Check the duality of the open-ended questions in the storybook JSON
    """
    duality_check_output = "./question_integration_debug"
    if not os.path.exists(duality_check_output):
        os.makedirs(duality_check_output)
    
    educare_storybooks = ["ada_twist_scientist", "a_letter_to_amy", "boxitects", "grumpy_monkey", "if_you_give_a_mouse_a_cookie", "jabari_jumps", "last_stop_on_market_street", "mango_abuela_and_me", "peters_chair", "stand_tall_molly_lou_melon", "the_proudest_blue"]

    kipp_storybooks = ["chicka_chicka_boom_boom", "the_little_red_hen-muldrow", "rap_a_tap_tap_heres_bojangles",
                       "three_little_pigs-marshall", "the_three_billy_goats_gruff-finch", "ganeshas_sweet_tooth",
                       "moon_rope", "the_story_of_ferdinand", "helpers_in_my_community", "a_day_in_the_life_of_a_firefighter",
                      "my_five_senses-aliki"]
    stories = kipp_storybooks + educare_storybooks

    checker = suitability_checker.GPTPedagogicQSuitabilityChecker()
    story_without_page_questions = []
    for story in stories:
        if os.path.exists(os.path.join(duality_check_output, story + ".json")):
            story_level_log = json.load(open(os.path.join(duality_check_output, story + ".json"), "r"))
            _remove_duplicate_prompts(os.path.join(qna_json_dir, story), story_level_log)
            continue

        print("...Processing story " + story)
        pages = sorted(list(filter(lambda x: x.endswith(".json"), os.listdir(os.path.join(qna_json_dir, story)))))
        story_level_log = dict()
        for page in pages[1:]:
            print("...Processing page " + page)
            with open(os.path.join(qna_json_dir, story, page), "r") as f:
                story_json = json.load(f)
            
            page_level_prompts = []
            checkpoint_prompt = None
            for prompt in story_json["prompts"]:
                if prompt["type"] == 5 and prompt["question_type"] == "open-ended":
                    if not prompt["is_checkpoint_question"]:
                        page_level_prompts.append(prompt)
                    else:
                        print("Checkpoint prompt: ", prompt["question"])
                        checkpoint_prompt = prompt
            if checkpoint_prompt is None:
                continue

            failed_uniqueness_check = 0
            for prompt in page_level_prompts:
                is_unique, explanation = checker._check_uniqueness_between_two_prompts(" ".join(story_json["text"]), checkpoint_prompt["question"], prompt["question"])
                if is_unique == False:
                    if page not in story_level_log.keys():
                        story_level_log[page] = []
                    story_level_log[page].append({"prompt": prompt["question"], "explanation": explanation})
                    failed_uniqueness_check += 1
            if failed_uniqueness_check == len(page_level_prompts):
                story_without_page_questions.append(page)
            
            with open(os.path.join(duality_check_output, story + ".json"), "w") as f:
                json.dump(story_level_log, f, indent=4)
    print("Story without page questions: ", '\n'.join(story_without_page_questions))
    

def main_integrate_questions_into_storybook_json(qna_json_dir):
    """
    Main function to integrate questions into storybook JSON
    """
    page_question_dir = "./outputs_q/rep_checked_ContextQ"
    moral_question_dir = "./outputs_q_moral/story_moral_q_2"

    integration_process = {
        "page_questions": False,
        "moral_questions": False,
        "check_duality": True
    }

    if integration_process["page_questions"]:
        page_question_stories = list(map(lambda x: x.replace(".json", ""), filter(lambda x: x.endswith(".json") and not x.endswith("_suitability_history.json"), os.listdir(page_question_dir))))
        for story in page_question_stories:
            integrate_page_questions_into_storybook(os.path.join(page_question_dir, story + ".json"), os.path.join(qna_json_dir, story))

    if integration_process["moral_questions"]:
        moral_question_stories = list(map(lambda x: x.replace(".json", ""), filter(lambda x: x.endswith(".json"), os.listdir(moral_question_dir))))
        for story in moral_question_stories:
            integrate_moral_questions_into_storybook_json(story, moral_question_dir, qna_json_dir)

    if integration_process["check_duality"]:
        check_duality_of_storybook_json_open_ended_questions(qna_json_dir)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main_integrate_questions_into_storybook_json(sys.argv[1])
    else:
        print("Usage: python ./question_integration.py <storybook_qna_path>")