import sys, os
from openai import OpenAI
import time
import pandas as pd
from pprint import pprint
import json
from sentence_transformers import SentenceTransformer
from collab_discourse_responses import ChitchatDiscourseAct, ChitchatResponse, PrspSharingDiscourseAct, PrspSharingResponse, CollaborativeDiscourseAct, CollaborativeDiscourseResponse

class CollaborativeDiscourseAgent:

    def __init__(self):
        self.client = OpenAI()
        self.temperature = .7
        self.frequency_panelty = 0.5
        self.presence_panelty = 0.5
        # self.model = "gpt-4.1-mini-2025-04-14"
        self.model = "gpt-4.1-2025-04-14"

        self.MAXIMUM_TRIES = 3

        self.prsp_sharing_prompt = self._load_prompt("./collab_discourse_prompts/prsp_sharing.txt")
        self.collab_discourse_prompt = self._load_prompt("./collab_discourse_prompts/collab_discourse.txt")

        self.chitchat_prompts = {
            "favoriate_book": self._load_prompt("./chitchat_prompts/chitchat_1_favoriate_book.txt"),
            "favoriate_season": self._load_prompt("./chitchat_prompts/chitchat_2_favoriate_season.txt"),
            "favoriate_activity": self._load_prompt("./chitchat_prompts/chitchat_3_favoriate_activity.txt")
        }

    def _load_prompt(self, prompt_file):
        with open(os.path.join(os.path.dirname(__file__), "prompts", prompt_file), "r") as f:
            return f.read()
    
    def _prsp_sharing_llm_call(self, prompt, current_text, dialog_history=[]):

        messages = [
            {"role": "system", "content": self.prsp_sharing_prompt.replace("$PROMPT$", prompt).replace("$CURRENT_TEXT$", current_text)},
        ] + dialog_history

        response = self.client.responses.parse(
            model=self.model,
            input=messages,
            text_format=PrspSharingResponse,
            temperature=self.temperature,
            top_p=1
        )

        return response.output_parsed
    
    def generate_prsp_sharing_action(self, prompt, current_text, dialog_history=[]):

        for i in range(self.MAXIMUM_TRIES):
            finished_prsp_sharing = False
            response = None
            try:
                response = self._prsp_sharing_llm_call(prompt, current_text, dialog_history)
            except:
                print("Error generating prsp sharing response, RETRY...", response)
                continue

            if response.discourse_act in [PrspSharingDiscourseAct.CURIOSITY, PrspSharingDiscourseAct.SUPPORT_UNDERSTANDING, PrspSharingDiscourseAct.QUESTION_BREAKDOWN]:
                finished_prsp_sharing = False
            elif response.discourse_act in [PrspSharingDiscourseAct.CREATIVE_IDEATION, PrspSharingDiscourseAct.ALTERNATIVE_PERSPECTIVE, PrspSharingDiscourseAct.OPPOSING_VIEW]:
                if response.robot_viewpoint == "" or response.child_viewpoint == "":
                    print("Robot/Child viewpoint is empty, RETRY...", response)
                    continue
                finished_prsp_sharing = True

            return finished_prsp_sharing, response
        
        return False, None

    def _collab_discourse_llm_call(self, prompt, current_text, robot_viewpoint, child_viewpoint, dialog_history=[]):

        messages = [
            {"role": "system", "content": self.collab_discourse_prompt.replace("$PROMPT$", prompt).replace("$CURRENT_TEXT$", current_text)},
        ] + dialog_history
        messages[0]["content"] = messages[0]["content"].replace("$ROBOT_VIEWPOINT$", robot_viewpoint).replace("$CHILD_VIEWPOINT$", child_viewpoint)

        response = self.client.responses.parse(
            model=self.model,
            input=messages,
            text_format=CollaborativeDiscourseResponse,
            temperature=self.temperature,
            top_p=1
        )

        return response.output_parsed
    
    def generate_collab_discourse_action(self, prompt, current_text, robot_viewpoint, child_viewpoint, dialog_history=[]):
        
        for i in range(self.MAXIMUM_TRIES):
            finished_discussion = False
            response = None
            try:
                response = self._collab_discourse_llm_call(prompt, current_text, robot_viewpoint, child_viewpoint, dialog_history)
            except:
                print("Error generating collaborative discourse response, RETRY...", response)
                continue
            
            if response.discourse_act == CollaborativeDiscourseAct.FINISHED_DISCUSSION:
                finished_discussion = True
            return finished_discussion, response
        
        return False, None
    
    def _chitchat_llm_call(self, session_topic, prior_knowledge="", dialog_history=[]):

        try:
            chitchat_prompt = self.chitchat_prompts[session_topic]
        except:
            print("Invalid session topic, using default prompt for favoriate book...")
            chitchat_prompt = self.chitchat_prompts["favoriate_book"]

        messages = [
            {"role": "system", "content": chitchat_prompt.replace("$PRIOR_KNOWLEDGE$", prior_knowledge)},
        ] + dialog_history

        response = self.client.responses.parse(
            model=self.model,
            input=messages,
            text_format=ChitchatResponse,
            temperature=self.temperature,
            top_p=1
        )

        return response.output_parsed
    
    def generate_chitchat_action(self, topic, prior_knowledge="", dialog_history=[]):

        for i in range(self.MAXIMUM_TRIES):
            finished_chitchat = False
            chitchat_response = None
            try:
                chitchat_response = self._chitchat_llm_call(topic, prior_knowledge, dialog_history)
            except:
                print("Error generating chitchat, RETRY...")
                continue

            if chitchat_response.discourse_act == ChitchatDiscourseAct.FINISHED_CHITCHAT:
                finished_chitchat = True
            
            return finished_chitchat, chitchat_response
        
        return False, None


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

def chitchat_interaction_testing(topic="", prior_knowledge=[], dialog_history=[]):
    collab_agent = CollaborativeDiscourseAgent()
    for turns in range(10):
        finished_chitchat, robot_response = collab_agent.generate_chitchat_action(topic, "\n".join(prior_knowledge), dialog_history)
        if robot_response is None:
            print("Error generating chitchat, FINISHING CHAT...")
            break

        print("___________________________")
        print("Discourse Act: ", robot_response.discourse_act)
        print("Robot Utterance: ", robot_response.robot_utterance)
        print("Child Answer: ", robot_response.child_answer)
        print("___________________________")
        if finished_chitchat:
            return robot_response.child_answer
        
        dialog_history.append({"role": "assistant", "content": robot_response.robot_utterance})
        child_response = generate_user_utterance("user")
        dialog_history.append({"role": "user", "content": child_response})
    
    return None

def test_all_chitchat_interactions():
    prior_knowledge = []
    for topic in ["favoriate_book", "favoriate_season", "favoriate_activity"]:
        print("___________________________")
        print("Topic: ", topic)
        print("___________________________")
        dialog_history = []
        prior = chitchat_interaction_testing(topic, prior_knowledge, dialog_history)
        if prior is not None and prior != "":
            prior_knowledge.append(prior)


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
            response = None
            discussion_finished = False
            discourse_act = None
            if discourse_step == "prsp_sharing":
                finished_prsp_sharing, response = collab_agent.generate_prsp_sharing_action(prompt, current_text, dialog_history)
                robot_utterance = response.robot_utterance
                discourse_act = response.discourse_act
                if finished_prsp_sharing:
                    robot_viewpoint = response.robot_viewpoint
                    child_viewpoint = response.child_viewpoint
                    print("****Summary of Child's View****")
                    print(child_viewpoint)
                    print("****Summary of Robot's View****")
                    print(robot_viewpoint)
                    print("****End of Summary*****")
                    discourse_step = "collab_discourse"
            elif discourse_step == "collab_discourse":
                discussion_finished, response = collab_agent.generate_collab_discourse_action(prompt, current_text, robot_viewpoint, child_viewpoint, dialog_history)
                robot_utterance = response.robot_utterance
                discourse_act = response.discourse_act
                print("___________________________")
                print("****Start of Collaborative Discourse****")
                print("___________________________")
            
            print("___________________________")
            print("Discourse Act: ", discourse_act)
            print("Response: ", robot_utterance)
            print("___________________________")
            if discussion_finished:
                break

            dialog_history.append({"role": "assistant", "content": robot_utterance})
            child_response = generate_user_utterance("user")
            dialog_history.append({"role": "user", "content": child_response})

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # main(sys.argv[1])
        # discourse_interaction_testing()
        chitchat_interaction_testing()
    else:
        print("Usage: ./gpt-interactive_discourse-testing.py")
        exit()