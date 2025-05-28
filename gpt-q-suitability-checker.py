import sys, os
from openai import OpenAI
import time
import pandas as pd
from pprint import pprint
import json
import base64
from sentence_transformers import SentenceTransformer

class GPTPedagogicQSuitabilityChecker:
    def __init__(self):
        self.client = OpenAI()
        self.model = "o4-mini-2025-04-16"
        self.temperature = .7
        self.top_p = .5

        self.suitability = {
            "open_ended": self._loadPrompt("prompts/suitability_prompts/ContextQ_suitability_check_open_ended.txt"),
            "authenticity": self._loadPrompt("prompts/suitability_prompts/ContextQ_suitability_check_authenticity.txt"),
            "relevance": self._loadPrompt("prompts/suitability_prompts/ContextQ_suitability_check_relevance.txt"),
            "uniqueness": self._loadPrompt("prompts/suitability_prompts/suitability_check_uniqueness.txt")
        }

    def _loadPrompt(self, prompt_file):
        f = open(prompt_file)
        prompt = f.read()
        return prompt
    
    def _completion(self, message):
        messages = [{"role": "system", "content": message}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # temperature=self.temperature,
            # max_tokens=500,
            # top_p=self.top_p
        )
        return response.choices[0].message.content
    
    def _check_suitability(self, suitability_type, prior_context, current_page, prompt):
        if suitability_type not in self.suitability.keys():
            raise ValueError("Invalid suitability type: ", suitability_type)
        
        system_prompt = self.suitability[suitability_type].replace("$PREVIOUS_PAGES$", prior_context).replace("$CURRENT_PAGE$", current_page).replace("$PROMPT$", prompt)
        for i in range(0, 3):
            try:
                response = json.loads(self._completion(system_prompt))
                return response[suitability_type], response
            except:
                print("Error in checking suitability for ", suitability_type, "trying again.")
        return False, None
    
    def _check_question_repeated(self, existing_questions, candidate_prompt):
        """
        Check if the candidate prompt is similar to any of the existing questions
        """
        if len(existing_questions) == 0:
            return False, None
        embedding_model = SentenceTransformer("all-mpnet-base-v2")
        embeddings_1 = embedding_model.encode(existing_questions)
        embeddings_2 = embedding_model.encode([candidate_prompt])
        similarities = embedding_model.similarity(embeddings_1, embeddings_2).flatten().tolist()
        max_sim = max(similarities)
        if max_sim >= 0.85:
            return True, existing_questions[similarities.index(max_sim)]
        else:
            return False, None
    
    def _check_uniqueness_between_two_prompts(self, current_text, candidate_prompt, other_candidate_prompt):
        """
        Check if the candidate prompt is unique
        Returns:
            is_unique: bool (True if unique, False otherwise)
            explanation: str
        """
        system_prompt = self.suitability["uniqueness"].replace("$CURRENT_PAGE$", current_text).replace("$PROMPT1$", candidate_prompt).replace("$PROMPT2$", other_candidate_prompt)
        for i in range(0, 3):
            try:
                response = json.loads(self._completion(system_prompt))
                if response["uniqueness"] == True:
                    return True, None
                else:
                    return False, "The prompt \"{}\" failed the uniqueness check because it is similar to \"{}\".".format(candidate_prompt, other_candidate_prompt)
            except:
                print("Error in checking uniqueness of prompts, trying again.")
        return False, None
    
    def check_uniqueness(self, current_text, candidate_prompt, existing_questions, other_candidate_prompt=[]):
        if len(other_candidate_prompt) > 0:
            for other_prompt in other_candidate_prompt:
                is_unique, explanation = self._check_uniqueness_between_two_prompts(current_text, candidate_prompt, other_prompt)
                if is_unique == False:
                    return False, explanation

        is_similar, similar_question = self._check_question_repeated(existing_questions, candidate_prompt)
        if is_similar == True:
            is_unique, explanation = self._check_uniqueness_between_two_prompts(current_text, candidate_prompt, similar_question)
            return is_unique, explanation
        else:
            return True, None
    
    def check_authenticity(self, prior_context, current_page, prompt):
        return self._check_suitability("authenticity", prior_context, current_page, prompt)
    
    def check_relevance(self, prior_context, current_page, prompt):
        return self._check_suitability("relevance", prior_context, current_page, prompt)
    
    def check_open_ended(self, prior_context, current_page, prompt):
        return self._check_suitability("open-ended", prior_context, current_page, prompt)
    
    def check_all_suitability(self, prior_context, current_page, prompt):
        results = {
            "prompt": prompt,
            "passed": True,
        }
        for suitability_type in ["open_ended", "authenticity", "relevance"]:
            suitable, explanation = self._check_suitability(suitability_type, prior_context, current_page, prompt)
            results[suitability_type] = {
                "explanation": explanation,
                "suitability": suitable
            }
            if not suitable:
                results["passed"] = False
                # print("Prompt: ", prompt, " failed the suitability check for ", suitability_type, " with explanation: ", explanation)
                return False, results
        # print("All suitability checks passed for prompt: ", prompt)
        return True, results
    
    def get_regeneration_prompt(self, results):
        regeneration_prompt = "The prompt, \"{}\", is not suitable for the following reasons: \n".format(results["prompt"])
        for suitability_type in ["open_ended", "authenticity", "relevance"]:
            if suitability_type in results.keys() and not results[suitability_type]["suitability"]:
                regeneration_prompt += "{}: {}\n".format(suitability_type, results[suitability_type]["explanation"])
        return regeneration_prompt
