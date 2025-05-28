import sys, os
from openai import OpenAI
import time
import pandas as pd
from pprint import pprint
import json
import base64
from sentence_transformers import SentenceTransformer
suitability_checker = importlib.import_module("gpt-q-suitability-checker")

class StorybookQnA:
    
    def __init__(self):
        # Setting the API key to use the OpenAI API
        # with open("/Users/xiajie/Projects/github-repos/openai-playground/collaborative_learning_api_key.txt", "r") as f:
        #     api_key = f.read()
        # print(api_key)
        self.client = OpenAI()
        self.temperature = .7
        self.top_p = .8

        self.model = "gpt-4.5-preview-2025-02-27"

        self.generator_prompt = self._loadPrompt("prompts/q_prompts_ContextQ/ContextQ_question_generator.txt")
        self.generator_prompt_image = self._loadPrompt("prompts/q_prompts_ContextQ/ContextQ_question_generator_image.txt")

    def _loadPrompt(self, prompt_file):
        f = open(prompt_file)
        prompt = f.read()
        return prompt
    
    def _assemble_system_prompt(self, storybook_context, type):
        if type == "generator":
            prompt = self.generator_prompt
        elif type == "generator_image":
            prompt = self.generator_prompt_image
        else:
            raise Exception("unrecognized prompt type: ", type)

        prompt = prompt.replace("$PREVIOUS_PAGE$", storybook_context['previous_page']).replace("$CURRENT_PAGE$", storybook_context['current_page'])
        if storybook_context["failed_attempts"] != "":
            regeneration_prompt = storybook_context["failed_attempts"] + " \nPlease generate another prompt that does not have the same issues."
        else:
            regeneration_prompt = ""
        prompt = prompt.replace("$REGENERATION_PROMPT$", regeneration_prompt)

        return prompt
    
    def _completion(self, message):
        messages = [{"role": "system", "content": message}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p
        )
        return response.choices[0].message.content
    
    def _completion_with_image(self, message, image_data):
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return response.choices[0].message.content

    def run_question_generator_with_image(self, storybook_context, image_path):

        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        
        systemprompt = self._assemble_system_prompt(storybook_context, "generator_image")
        json_response = json.loads(self._completion_with_image(systemprompt, encode_image(image_path)))

        # Decipher response
        try:
            generated_prompt = json_response["prompt"]
        except:
            raise Exception("Something went wrong generating prompt: ", json_response)
        return generated_prompt

    def run_question_generator(self, storybook_context):
        systemprompt = self._assemble_system_prompt(storybook_context, "generator")
        # print("System Prompt: ", systemprompt)
        json_response = json.loads(self._completion(systemprompt))
        
        # Decipher response
        try:
            generated_prompt = json_response["prompt"]
        except:
            raise Exception("Something went wrong generating prompt: ", json_response)
        return generated_prompt
    
    def _increase_randomness_llm_param(self):
        self.temperature = .9
    
    def _reset_llm_param(self):
        self.temperature = .7

def get_image_path(qna_path):

    assert qna_path.endswith(".json"), "qna path does not have the format: " + qna_path

    image_path = qna_path[:-5].replace("qna_json", "image") + "/Background.png"
    # print(image_path)
    assert os.path.isfile(image_path), "image does not exist: " + image_path
    return image_path

def main(storybook_qna_dir):

    output_dir = "outputs_q/rep_checked_ContextQ"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    qa = StorybookQnA()
    checker = suitability_checker.SuitabilityChecker()
    generateWithImage = False
    if generateWithImage:
        save_prefix = "image_"
    else:
        save_prefix = ""

    educare_storybooks = ["ada_twist_scientist", "a_letter_to_amy", "boxitects", "grumpy_monkey", "if_you_give_a_mouse_a_cookie", \
                          "jabari_jumps", "last_stop_on_market_street", "mango_abuela_and_me", "peters_chair", "stand_tall_molly_lou_melon", "the_proudest_blue"]

    kipp_storybooks = ["chicka_chicka_boom_boom", "the_little_red_hen-muldrow", "rap_a_tap_tap_heres_bojangles",
                       "three_little_pigs-marshall", "the_three_billy_goats_gruff-finch", "ganeshas_sweet_tooth",
                       "moon_rope", "the_story_of_ferdinand", "helpers_in_my_community", "a_day_in_the_life_of_a_firefighter",
                      "my_five_senses-aliki"]
    stories = educare_storybooks + kipp_storybooks

    for storybook in stories:
        storybook_path = os.path.join(storybook_qna_dir, storybook)
        story_output_path = os.path.join(output_dir, save_prefix + storybook + ".json")
        story_suitability_output_path = os.path.join(output_dir, save_prefix + storybook + "_suitability_history.json")
        if os.path.exists(story_output_path):
            with open(story_output_path, "r") as f:
                log_generated_q = json.load(f)
        else:
            log_generated_q = dict()
        
        if os.path.exists(story_suitability_output_path):
            with open(story_suitability_output_path, "r") as f:
                log_suitability = json.load(f)
        else:
            log_suitability = dict()
        
        all_pages = sorted(list(filter(lambda x: x.endswith('.json'), os.listdir(storybook_path))))
        # print(all_pages[1:])
        print("\n\nProcessing storybook ", storybook)

        previous_pages = []
        previous_prompts = []
        question_per_page = 2
        for page in all_pages[1:]:

            with open(os.path.join(storybook_path, page), "r") as f:
                content = json.load(f)
            current_page = ' '.join(content['text'])

            if page in log_generated_q.keys() and len(log_generated_q[page]) >= question_per_page:
                # skip the page if it already has the required number of prompts, but update the previous prompts for later use
                previous_pages.append(current_page)
                previous_prompts += [q["prompt"] for q in log_generated_q[page]]
                continue

            storybook_context = {
                "current_page": current_page,
                "previous_page": previous_pages[-1] if len(previous_pages) > 0 else "",
                "prior_context": ' '.join(previous_pages),
            }

            log_generated_q[page] = []
            log_suitability[page] = []
            prompt = ""
            failed_attempts = []
            qa.reset_llm_param()
            # A five times loop if prompt failed
            for attempt in range(0, 6):
                if len(failed_attempts) > 0:
                    print("...attempting again ", attempt)
                    if attempt > 2:
                        # Temporarially bump up the randomness of llm generation for the last few attempts
                        qa.increase_randomness_llm_param()

                # generator
                if generateWithImage:
                    prompt = qa.run_generator_with_image(storybook_context, get_image_path(os.path.join(storybook_path, page)))
                else:
                    storybook_context["failed_attempts"] = "\n".join(failed_attempts)
                    prompt = qa.run_generator(storybook_context)

                storybook_context["prompt"] = prompt

                # Step 2: Check the uniqueness of the prompt
                is_unique, explanation = checker.check_uniqueness(storybook_context["current_page"], prompt, previous_prompts)
                if not is_unique:
                    if explanation is not None:
                        failed_attempts.append(explanation)
                    log_suitability[page].append({"prompt": prompt, "passed": False, "uniqueness": explanation})
                    continue
                
                # Step 3: Check the suitability of the prompt
                passed, results = checker.check_all_suitability(storybook_context["prior_context"], storybook_context["current_page"], prompt)
                log_suitability[page].append(results)
                if passed:
                    log_generated_q[page].append({
                        "prompt": prompt,
                        "current_page": current_page
                    })
                    previous_prompts.append(prompt)
                else:
                    failed_attempts.append(checker.get_regeneration_prompt(results))
                
                if len(log_generated_q[page]) >= question_per_page:
                    break
            
            if len(log_generated_q[page]) < question_per_page:
                print("*"*30)
                print("No enough prompt was generated for page: ", page, "with ", len(log_generated_q[page]), "prompts.")
            
            previous_pages.append(current_page)
            with open(story_output_path, 'w') as f:
                json.dump(log_generated_q, f, indent=4)
            with open(story_suitability_output_path, "w") as f:
                json.dump(log_suitability, f, indent=4)
            
            print("Finished page ", page)
            

if __name__ == "__main__":
    if len(sys.argv) == 2:
         main(sys.argv[1])
    else:
         print ("Usage: ./gpt-q-gen-ContextQ.py <storybook_qna_folder>")
         exit()

