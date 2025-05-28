"""
This script is used to generate student responses for a given storybook prompt.
"""
import json
import os
import sys
from openai import OpenAI
from pprint import pprint

class StudentResponseSim:
    def __init__(self, student_type="model_student"):
        self.client = OpenAI()
        self.model = "gpt-4.5-preview-2025-02-27"
        self.temperature = 0.7
        if student_type in ["model_student", "passive_student"]:
            self.student_type = student_type
        else:
            # Default to model_student
            self.student_type = "model_student"
        self.prompt = self._load_prompt(os.path.join("./prompts/student_sim_prompt", self.student_type + "_student.txt"))

    def _load_prompt(self, prompt_file):
        with open(prompt_file, "r") as f:
            return f.read()
    
    def generate_response(self, story_context, question):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.prompt.replace("$STORY_CONTEXT$", story_context).replace("$QUESTION$", question)}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content
    
    def sample_responses(self, story_context, question, num_samples=1):
        responses = []
        max_retries = num_samples * 2
        while len(responses) < num_samples:
            try:
                response_candidate = json.loads(self.generate_response(story_context, question))
                responses.append({
                    "response": response_candidate["response"],
                    "reasoning": response_candidate["reasoning"]
                })
                if len(responses) == num_samples:
                    break
            except Exception as e:
                max_retries -= 1
                if max_retries <= 0:
                    break
        return responses, len(responses) == num_samples
        
