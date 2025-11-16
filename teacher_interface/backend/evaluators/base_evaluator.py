from __future__ import annotations

from typing import Any, Dict, Optional


class BaseEvaluator:
    """Abstract base class for suitability evaluators."""

    def __init__(
        self,
        name: str,
        description: str,
        rubric: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.rubric = rubric or {}
        self.weight = weight
        self.metadata = metadata or {}

    def evaluate(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate a question and return structured scoring details."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "rubric": self.rubric,
            "weight": self.weight,
        }

