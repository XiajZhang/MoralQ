from __future__ import annotations

import re
from typing import Any, Dict, Optional

import dspy

from .base_evaluator import BaseEvaluator


def snake_to_camel(value: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", value)
    return "".join(part.capitalize() for part in parts if part)


def make_dynamic_signature(evaluator_name: str, guidance: str) -> type:
    class_name = f"{snake_to_camel(evaluator_name)}DynamicSignature"
    attrs = {
        "__doc__": f"Dynamic evaluator signature for {evaluator_name}",
        "question": dspy.InputField(desc="Question to evaluate"),
        "story_context": dspy.InputField(desc="Story context or passage summary"),
        "objective": dspy.InputField(desc="Lesson objective or evaluator focus"),
        "guidance": dspy.InputField(desc="Evaluator guidelines or prompt"),
        "score": dspy.OutputField(desc="Score from 1-5 representing evaluator alignment"),
        "rationale": dspy.OutputField(desc="Brief explanation of the score"),
    }
    return type(class_name, (dspy.Signature,), attrs)


def make_dynamic_evaluator_class(
    evaluator_name: str,
    description: str,
    metadata: Dict[str, Any],
    agent: dspy.Predict,
) -> type:
    guidance = metadata.get("prompt") or metadata.get("instruction") or description
    default_weight = metadata.get("default_weight", 0.1)
    rubric = metadata.get("rubric", {})

    class_name = f"{snake_to_camel(evaluator_name)}DynamicEvaluator"

    class GeneratedDynamicEvaluator(BaseEvaluator):
        def __init__(
            self,
            name: str = evaluator_name,
            description: str = description,
            rubric: Optional[Dict[str, Any]] = None,
            weight: float = default_weight,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> None:
            merged_metadata = metadata or {}
            merged_metadata.setdefault("prompt", guidance)
            merged_metadata.setdefault("default_weight", default_weight)
            super().__init__(
                name=name,
                description=description,
                rubric=rubric or rubric,
                weight=weight,
                metadata=merged_metadata,
            )
            self._agent = agent
            self._guidance = guidance

        def evaluate(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            context = context or {}
            story_context = context.get("story", "")
            objective = context.get("objective", "")

            try:
                result = self._agent(
                    question=question,
                    story_context=story_context,
                    objective=objective,
                    guidance=self._guidance,
                )
                score_raw = getattr(result, "score", 3.0)
                score = float(score_raw)
                reasoning = getattr(result, "rationale", getattr(result, "reasoning", ""))
                return {
                    "score": score,
                    "reasoning": reasoning or "LLM evaluator rationale not provided.",
                }
            except Exception as exc:
                return {
                    "score": 3.0,
                    "reasoning": f"Default score due to evaluator error: {exc}",
                }

    GeneratedDynamicEvaluator.__name__ = class_name
    return GeneratedDynamicEvaluator

