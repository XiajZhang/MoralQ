from pydantic import BaseModel
from enum import Enum
import sys, os
from openai import OpenAI

class ChitchatDiscourseAct(Enum):
    INITIATION = "INITIATION"
    CURIOSITY = "CURIOSITY"
    SHARING = "SHARING"
    FINISHED_CHITCHAT = "FINISHED_CHITCHAT"

class PrspSharingDiscourseAct(Enum):
    CURIOSITY = "CURIOSITY"
    SUPPORT_UNDERSTANDING = "SUPPORT_UNDERSTANDING"
    QUESTION_BREAKDOWN = "QUESTION_BREAKDOWN"
    CREATIVE_IDEATION = "CREATIVE_IDEATION"
    ALTERNATIVE_PERSPECTIVE = "ALTERNATIVE_PERSPECTIVE"
    OPPOSING_VIEW = "OPPOSING_VIEW"

class CollaborativeDiscourseAct(Enum):
    ASKING_ELABORATION = "ASKING_ELABORATION"
    PROVIDE_ELABORATION = "PROVIDE_ELABORATION"
    BUILDING_ON = "BUILDING_ON"
    FINISHED_DISCUSSION = "FINISHED_DISCUSSION"

class ChitchatResponse(BaseModel):
    discourse_act: ChitchatDiscourseAct
    robot_utterance: str
    child_answer: str

class PrspSharingResponse(BaseModel):
    discourse_act: PrspSharingDiscourseAct
    robot_utterance: str
    child_viewpoint: str
    robot_viewpoint: str

class CollaborativeDiscourseResponse(BaseModel):
    discourse_act: CollaborativeDiscourseAct
    robot_utterance: str