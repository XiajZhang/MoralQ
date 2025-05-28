import sys, os
from openai import OpenAI
import time
import pandas as pd
from pprint import pprint
import json
from sentence_transformers import SentenceTransformer

class CollaborativeDiscourseAgent:

    def __init__(self):
        self.client = OpenAI()
        self.temperature = .7
        self.frequency_panelty = 0.5
        self.presence_panelty = 0.5
        # self.model = "gpt-4.1-mini-2025-04-14"
        self.model = "gpt-4.1-2025-04-14"

        self.prsp_sharing_prompt = self._load_prompt("./prompts/collab_discourse_prompts/prsp_sharing.txt")
        self.collab_discourse_prompt = self._load_prompt("./prompts/collab_discourse_prompts/collab_discourse.txt")

    def _load_prompt(self, prompt_file):
        with open(prompt_file, "r") as f:
            return f.read()
    
    def _prsp_sharing_llm_call(self, prompt, current_text, previous_context, dialog_history=[]):

        messages = [
            {"role": "system", "content": self.prsp_sharing_prompt.replace("$PROMPT$", prompt).replace("$CURRENT_TEXT$", current_text).replace("$PREVIOUS_CONTEXT$", previous_context)},
        ] + dialog_history

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            # max_tokens=1024,
            top_p=1,
            frequency_penalty=self.frequency_panelty,
            presence_penalty=self.presence_panelty
        )

        return response.choices[0].message.content
    
    def generate_prsp_sharing_action(self, prompt, current_text, previous_context, dialog_history=[]):

        ERROR_RETRIES = 3
        response = None
        for i in range(ERROR_RETRIES):
            finished_prsp_sharing = False
            response = self._prsp_sharing_llm_call(prompt, current_text, previous_context, dialog_history)
            try:
                json_response = json.loads(response)
            except:
                print("Error parsing response, RETRY...", response.choices[0].message.content)
                continue
            
            # Checking formats
            if "discourse_act" not in json_response.keys() or "robot_utterance" not in json_response.keys():
                print("Missing values in response, RETRY...", response.choices[0].message.content)
                continue

            if json_response["discourse_act"] in ["CURIOSITY", "SUPPORT_UNDERSTANDING", "QUESTION_BREAKDOWN"]:
                finished_prsp_sharing = False
                response = json_response
            elif json_response["discourse_act"] in ["CREATIVE_IDEATION", "ALTERNATIVE_PERSPECTIVE", "OPPOSING_VIEW"]:
                if json_response["robot_viewpoint"] == "":
                    print("Robot viewpoint is empty, RETRY...", response.choices[0].message.content)
                    continue
                finished_prsp_sharing = True
                response = json_response
            else:
                print("Unrecognized discourse act: ", json_response["discourse_act"])
                continue
            return finished_prsp_sharing, response
        
        return False, None

    def generate_collab_discourse_action(self, prompt, current_text, previous_context, robot_viewpoint, child_viewpoint, dialog_history=[]):
        messages = [
            {"role": "system", "content": self.collab_discourse_prompt.replace("$PROMPT$", prompt).replace("$CURRENT_TEXT$", current_text).replace("$PREVIOUS_CONTEXT$", previous_context)},
        ] + dialog_history
        messages[0]["content"] = messages[0]["content"].replace("$ROBOT_VIEWPOINT$", robot_viewpoint).replace("$CHILD_VIEWPOINT$", child_viewpoint)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            # max_tokens=1024,
            top_p=1,
            frequency_penalty=self.frequency_panelty,
            presence_penalty=self.presence_panelty
        )

def generate_user_utterance(type="user"):
    user_response = ""
    if type == "user":
        human_input = input("Enter your response: ")
        if human_input == "q":
            user_response = ""
        else:
            user_response = human_input
    elif type == "sim":
        user_response = "The robot is thinking..."
    return user_response

def retrieve_perspectives(viewpoints, user_viewpoint):
    viewpoint_list = [x["view_point"] for x in viewpoints]
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    embeddings_1 = embedding_model.encode(viewpoint_list)
    embeddings_2 = embedding_model.encode([user_viewpoint])
    similarities = embedding_model.similarity(embeddings_1, embeddings_2).flatten().tolist()
    max_similarity_index = similarities.index(max(similarities))
    min_similarity_index = similarities.index(min(similarities))
    return viewpoints[max_similarity_index], viewpoints[min_similarity_index]

def discourse_interaction_testing(storybook_qna_dir=""):

    # Fix the storybook_qna_dir to be the absolute path
    storybook_qna_dir = "/Users/xiajie/Projects/github-repos/interactive-storybook-assets/qna_json/"

    collab_agent = CollaborativeDiscourseAgent()

    testing_book = "a_day_in_the_life_of_a_firefighter"
    book_pages = list(filter(lambda x: x.endswith(".json"), os.listdir(os.path.join(storybook_qna_dir, testing_book))))
    book_pages.sort()

    rolling_story_content = []
    for page in book_pages[1:]:
        page_path = os.path.join(storybook_qna_dir, testing_book, page)
        with open(page_path, "r") as f:
            page_data = json.load(f)

        current_text = " ".join(page_data["text"])
        rolling_story_content.append(current_text)
        prompt = list(filter(lambda x: x["type"] == 5 and x["question_type"] == "open-ended", page_data["prompts"]))[0]["question"]
        
        dialog_history = []
        print("___________________________")
        print("Question: ", prompt)
        print("___________________________")
        discourse_step = "prsp_sharing"
        child_viewpoint = None
        robot_viewpoint = None
        robot_utterance = ""
        for turns in range(10):
            response = {}
            if discourse_step == "prsp_sharing":
                prior_context = " ".join(rolling_story_content[:-5]) if len(rolling_story_content) > 5 else " ".join(rolling_story_content)
                finished_prsp_sharing, response = collab_agent.generate_prsp_sharing_action(prompt, current_text, prior_context, dialog_history)
                robot_utterance = response["robot_utterance"]
                if finished_prsp_sharing:
                    robot_viewpoint = response["robot_viewpoint"]
                    child_viewpoint = response["child_viewpoint"]
                    print("****Summary of Child's View****")
                    print(child_viewpoint)
                    print("****Summary of Robot's View****")
                    print(robot_viewpoint)
                    print("****End of Summary*****")
                    discourse_step = "collab_discourse"
            elif discourse_step == "collab_discourse":
                response = collab_agent.generate_collab_discourse_action(prompt, current_text, prior_context, robot_viewpoint, child_viewpoint, dialog_history)
                pprint(response)
                robot_utterance = response["robot_utterance"]
                print("___________________________")
                print("****Start of Collaborative Discourse****")
                print("___________________________")
            

            dialog_history.append({"role": "assistant", "content": robot_utterance})
            print("___________________________")
            print("Discourse Act: ", response["discourse_act"])
            print("Response: ", response["robot_utterance"])
            print("___________________________")

            child_response = generate_user_utterance("user")
            dialog_history.append({"role": "user", "content": child_response})

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # main(sys.argv[1])
        discourse_interaction_testing()
    else:
        print("Usage: ./gpt-interactive_discourse-testing.py")
        exit()