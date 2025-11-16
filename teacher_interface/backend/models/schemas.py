from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class EvaluatorMetadataModel(BaseModel):
    description: str
    rubric: Dict[str, Any] = Field(default_factory=dict)
    default_weight: float = Field(0.0, ge=0.0, le=1.0)
    module: str
    status: str = Field(default="active")

    class Config:
        extra = "allow"


class EvaluatorConfigModel(BaseModel):
    __root__: Dict[str, EvaluatorMetadataModel] = Field(default_factory=dict)

    def to_mapping(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: metadata.dict(exclude_none=True)
            for name, metadata in self.__root__.items()
        }

    def keys(self):
        return self.__root__.keys()

    def items(self):
        return self.__root__.items()


class FeedbackActionModel(BaseModel):
    type: str
    affected: List[str] = Field(default_factory=list)
    delta: Dict[str, float] = Field(default_factory=dict)
    reason: str = ""
    reformulated_instruction: str = ""
    details: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    class Config:
        extra = "allow"


class FeedbackRecordModel(BaseModel):
    story_title: str
    objective: str
    iteration_id: str
    teacher_feedback: Dict[str, Any]
    interpretation: Dict[str, Any]
    action_taken: FeedbackActionModel
    evaluator_scores: Dict[str, Union[int, float]] = Field(default_factory=dict)
    question_evaluations: List[Dict[str, Any]] = Field(default_factory=list)
    course_of_action: Dict[str, Any]
    timestamp: str
    teacher_id: str
    school_id: str
    story_context: Optional[str] = None
    generated_questions: List[Any] = Field(default_factory=list)
    question_feedbacks: Dict[Any, Any] = Field(default_factory=dict)
    routed_message: Optional[Dict[str, Any]] = None
    manager_response: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class EvaluationResultModel(BaseModel):
    question: str
    question_type: Optional[str] = None
    type_confidence: Optional[float] = None
    suitability_score: Optional[float] = None
    decision: Optional[str] = None
    evaluation_reasoning: Optional[str] = None
    dynamic_evaluations: Dict[str, Any] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class EvaluationLogModel(BaseModel):
    evaluations: List[EvaluationResultModel] = Field(default_factory=list)

    class Config:
        extra = "allow"

