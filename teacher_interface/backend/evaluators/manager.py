from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from .registry import create_evaluator as registry_create_evaluator
from .suitability_evaluators import (
    create_evaluator as create_evaluator_with_metadata,
    get_evaluator_metadata,
    list_registered_dynamic_evaluators,
    list_registered_evaluators,
    persist_dynamic_metadata,
)
from teacher_interface.backend.models import EvaluatorMetadataModel


class EvaluatorManager:
    """Central service for creating, weighting, and running evaluators."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        dynamic_path: Optional[str] = None,
        dynamic_creator: Optional[Any] = None,
    ) -> None:
        self.config_path = config_path
        self.dynamic_path = dynamic_path
        self.dynamic_creator = dynamic_creator

        self.evaluators: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}

        self._load_all()

    def refresh(self) -> None:
        """Reload evaluators and weights from stored metadata."""
        self.evaluators.clear()
        self.weights.clear()
        self._load_all()

    def _load_all(self) -> None:
        core_meta = list_registered_evaluators()
        dynamic_meta = list_registered_dynamic_evaluators()

        combined: Dict[str, Dict[str, Any]] = {}
        combined.update(core_meta)
        combined.update(dynamic_meta)

        for name, meta in combined.items():
            metadata = EvaluatorMetadataModel.parse_obj(meta).model_dump(exclude_none=True)
            # Create evaluator with metadata-aware factory to pass name/description/rubric/weight
            evaluator = create_evaluator_with_metadata(name)
            evaluator.weight = metadata.get("default_weight", evaluator.weight)
            self.evaluators[name] = evaluator
            self.weights[name] = metadata.get("default_weight", evaluator.weight)

        self._renormalize()

    def load_from_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        evaluators = {key: create_evaluator(key) for key in data.keys()}
        return evaluators

    def update_weight(self, name: str, delta: float, clamp: bool = True) -> None:
        current = self.weights.get(name)
        if current is None:
            return

        new_weight = current + delta
        if clamp:
            new_weight = max(0.05, min(0.4, new_weight))

        self.weights[name] = round(new_weight, 4)
        self._renormalize()

    def evaluate_all(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for name, evaluator in self.evaluators.items():
            try:
                results[name] = evaluator.evaluate(question, context)
            except NotImplementedError:
                results[name] = {
                    "score": None,
                    "reasoning": "Evaluator not implemented",
                }
        return results

    def process_message(self, message) -> Dict[str, Any]:
        response: Dict[str, Any] = {"details": None}

        if message.action == "reinforce_existing":
            self._reinforce_weights()
        elif message.action == "adjust_evaluator":
            for evaluator_name, delta in message.delta.items():
                self.update_weight(evaluator_name, delta)
        elif message.action == "add_new_evaluator":
            response["details"] = self._register_dynamic_evaluator(message.new_evaluator)

        response["weights"] = dict(self.weights)
        return response

    def _reinforce_weights(self, rate: float = 0.05) -> None:
        for name in list(self.weights.keys()):
            self.weights[name] = min(0.5, round(self.weights[name] * (1.0 + rate), 4))
        self._renormalize()

    def _renormalize(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            return
        for name in list(self.weights.keys()):
            self.weights[name] = round(self.weights[name] / total, 4)

    def renormalize(self) -> None:
        self._renormalize()

    def _register_dynamic_evaluator(self, spec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not spec or "name" not in spec:
            return None

        name = spec["name"]
        description = spec.get("description", f"Evaluates questions for {name.replace('_', ' ')}")
        weight = float(spec.get("weight", 0.1))
        rubric = spec.get("rubric", {})

        metadata_payload: Dict[str, Any] = {
            "description": description,
            "default_weight": weight,
            "rubric": rubric,
            "module": spec.get("module", "teacher_interface.backend.evaluators.suitability_evaluators"),
            "status": spec.get("status", "active"),
            "prompt": spec.get("prompt") or rubric.get("instruction", ""),
            "template": spec.get("template", "llm_dynamic"),
            "origin": spec.get("origin", {}),
            "created_at": spec.get("created_at", datetime.now().isoformat()),
        }

        metadata_model = EvaluatorMetadataModel.parse_obj(metadata_payload)
        metadata_payload = metadata_model.model_dump(exclude_none=True)
        metadata_payload["default_weight"] = metadata_payload.get("default_weight", weight)

        if self.dynamic_creator:
            try:
                created_metadata = self.dynamic_creator.create_new_evaluator({"name": name, **metadata_payload})
                if created_metadata:
                    metadata_payload.update(created_metadata)
            except Exception as exc:
                print(f"[WARN] Failed to instantiate dynamic evaluator '{name}': {exc}")
                persist_dynamic_metadata(name, metadata_payload)
        else:
            persist_dynamic_metadata(name, metadata_payload)

        self.refresh()

        metadata_payload["evaluator_name"] = name
        metadata_payload["weight"] = self.weights.get(name, weight)
        metadata_payload["weights_snapshot"] = dict(self.weights)
        return metadata_payload

