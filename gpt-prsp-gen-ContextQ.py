"""
This script is used to generate perspectives for a given storybook prompt.
"""

import json
import os
import sys
from openai import OpenAI
from pprint import pprint

class PerspectivesGeneration:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4.5-preview-2025-02-27"
        self.temperature = 0.7
        self.max_tokens = 1024
        self.top_p = 1
        self.presence_penalty = 0.5
        self.frequency_penalty = 0.5
        self.prompt = self.load_prompt("./prompts/prsp_prompts/diverse_perspective_sampling.txt")

    def load_prompt(self, prompt_file):
        with open(prompt_file, "r") as f:
            return f.read()
    
    def generate_perspectives(self, prompt, current_content):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.prompt.replace("$PROMPT$", prompt).replace("$CURRENT_CONTENT$", current_content)}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )
        return response.choices[0].message.content
    
def main():
    perspectives_generation = PerspectivesGeneration()

    storybook_prompt_dir = "./outputs"
    perspectives_output_dir = "./perspectives"
    if not os.path.exists(perspectives_output_dir):
        os.makedirs(perspectives_output_dir)
    storybooks = list(filter(lambda x: x.endswith("prompt.json"), os.listdir(storybook_prompt_dir)))
    for storybook_prompt in storybooks:
        print("Processing ", storybook_prompt.replace("_prompt.json", ""))
        
        if os.path.exists(os.path.join(perspectives_output_dir, storybook_prompt)):
            with open(os.path.join(perspectives_output_dir, storybook_prompt), "r") as f:
                storybook_info = json.load(f)
        else:
            with open(os.path.join(storybook_prompt_dir, storybook_prompt), "r") as f:
                storybook_info = json.load(f)

        for page in storybook_info.keys():
            # Check if the page has already been processed
            if "perspectives" in storybook_info[page].keys():
                assert len(storybook_info[page]["perspectives"]) > 2, "The number of perspectives for " + page + " is less than 2"
                assert len(storybook_info[page]["perspectives"]) < 7, "The number of perspectives for " + page + " is greater than 7" + str(len(storybook_info[page]["perspectives"]))
                continue

            print("....Page ", page.split("_")[-1].replace(".json", ""))
            prompt = storybook_info[page]["prompt"]
            current_content = storybook_info[page]["current_page"]
            for i in range(3):
                response = perspectives_generation.generate_perspectives(prompt, current_content)
                try:
                    perspectives = json.loads(response)
                    storybook_info[page]["perspectives"] = list(perspectives.values())
                    break
                except Exception as e:
                    if i == 2:
                        print("Error in generating perspectives for ", page, " with error ", e)
                        print("Response: ", response)
                        sys.exit()
                    else:
                        print("Retrying for ", page, " for the ", i+1, " time")
            # Save the storybook prompt with perspectives
            with open(os.path.join(perspectives_output_dir, storybook_prompt), "w") as f:
                json.dump(storybook_info, f, indent=4)

if __name__ == "__main__":
    main()
