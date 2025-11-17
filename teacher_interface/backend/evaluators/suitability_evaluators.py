from __future__ import annotations

import importlib
import json
import os
from typing import Any, Dict, Optional, Set

import dspy

from .base_evaluator import BaseEvaluator
from .registry import (
    list_core_evaluators,
    list_dynamic_evaluators,
    register_core_evaluator,
    register_dynamic_evaluator,
    set_fallback_evaluator,
    get_evaluator_class,
)
from teacher_interface.backend.models import EvaluatorConfigModel, EvaluatorMetadataModel
from .dynamic_template import make_dynamic_evaluator_class, make_dynamic_signature


EVALUATOR_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "evaluators.json",
)

DYNAMIC_EVALUATOR_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "dynamic_evaluators.json",
)

_core_metadata: Dict[str, Dict[str, Any]] = {}
_dynamic_metadata: Dict[str, Dict[str, Any]] = {}
_loaded_modules: Set[str] = set()


def _read_json_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle) or {}
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Failed to read evaluator config {path}: {exc}")
        return {}


def _write_json_file(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _ensure_config_files() -> None:
    if not os.path.exists(EVALUATOR_CONFIG_PATH):
        _write_json_file(EVALUATOR_CONFIG_PATH, {})
    if not os.path.exists(DYNAMIC_EVALUATOR_CONFIG_PATH):
        _write_json_file(DYNAMIC_EVALUATOR_CONFIG_PATH, {})


def _ensure_module_loaded(module_path: Optional[str]) -> None:
    if not module_path:
        return
    if module_path in _loaded_modules:
        return
    try:
        importlib.import_module(module_path)
        _loaded_modules.add(module_path)
    except ImportError as exc:
        print(f"[WARN] Failed to import evaluator module '{module_path}': {exc}")


def load_evaluator_metadata() -> None:
    global _core_metadata, _dynamic_metadata
    _ensure_config_files()
    core_raw = _read_json_file(EVALUATOR_CONFIG_PATH)
    dynamic_raw = _read_json_file(DYNAMIC_EVALUATOR_CONFIG_PATH)

    core_config = EvaluatorConfigModel.parse_obj(core_raw)
    dynamic_config = EvaluatorConfigModel.parse_obj(dynamic_raw)

    _core_metadata = core_config.to_mapping()
    _dynamic_metadata = dynamic_config.to_mapping()

    for name, metadata in _core_metadata.items():
        _ensure_module_loaded(metadata.get("module"))

    for name, metadata in _dynamic_metadata.items():
        _ensure_module_loaded(metadata.get("module"))
        template = metadata.get("template", "")
        if template == "llm_dynamic":
            prompt = metadata.get("prompt") or metadata.get("rubric", {}).get("instruction", "")
            description = metadata.get("description", name.replace("_", " "))
            signature = make_dynamic_signature(name, prompt)
            agent = dspy.Predict(signature)
            evaluator_cls = make_dynamic_evaluator_class(name, description, metadata, agent)
            register_dynamic_evaluator(name)(evaluator_cls)
        else:
            register_dynamic_evaluator(name)(DynamicEvaluator)


def get_evaluator_metadata(name: str) -> Dict[str, Any]:
    if name in _dynamic_metadata:
        return _dynamic_metadata[name]
    return _core_metadata.get(name, {})


def persist_dynamic_metadata(name: str, metadata: Dict[str, Any]) -> None:
    metadata_dict = dict(metadata)
    metadata_dict.setdefault("module", "teacher_interface.backend.evaluators.suitability_evaluators")
    meta_model = EvaluatorMetadataModel.parse_obj(metadata_dict)
    metadata_sanitized = meta_model.model_dump(exclude_none=True)

    _dynamic_metadata[name] = metadata_sanitized
    _ensure_module_loaded(metadata_sanitized.get("module"))
    register_dynamic_evaluator(name)(DynamicEvaluator)
    _write_json_file(DYNAMIC_EVALUATOR_CONFIG_PATH, _dynamic_metadata)


def create_evaluator(name: str) -> BaseEvaluator:
    metadata = get_evaluator_metadata(name)
    cls = get_evaluator_class(name)
    return cls(
        name=name,
        description=metadata.get("description", name.replace("_", " ")),
        rubric=metadata.get("rubric", {}),
        weight=metadata.get("default_weight", 1.0),
        metadata=metadata,
    )

#
# DSPy signatures for core suitability evaluators
#
class TypeClassificationSignature(dspy.Signature):
    """Classifies the question type based on the rubric."""
    question = dspy.InputField(desc="The question to classify")
    question_type = dspy.OutputField(desc="One of: Completion, Recall, Open-Ended, Wh, Distancing")
    confidence = dspy.OutputField(desc="High, Medium, or Low confidence in classification")
    reasoning = dspy.OutputField(desc="Brief explanation of the classification")


class CompletionSuitabilitySignature(dspy.Signature):
    """Evaluates suitability for Completion questions."""
    question = dspy.InputField(desc="The completion question")
    deals_with_rhyming_or_repeated_phrases = dspy.OutputField(desc="True/False: rhyming or repeated phrases?")
    suitability_score = dspy.OutputField(desc="Score from 0-1 (1 if criteria met, 0 otherwise)")
    decision = dspy.OutputField(desc="'pass' if criteria met, 'regenerate' otherwise")
    reasoning = dspy.OutputField(desc="Explanation of the evaluation")


class RecallSuitabilitySignature(dspy.Signature):
    """Evaluates suitability for Recall questions."""
    question = dspy.InputField(desc="The recall question")
    plot_elements_or_sequences = dspy.OutputField(desc="True/False: asks about plot elements or sequences?")
    answer_beyond_current_page = dspy.OutputField(desc="True/False: requires beyond current page?")
    suitability_score = dspy.OutputField(desc="Average of two subcriteria (0-1)")
    decision = dspy.OutputField(desc="'pass' if all criteria met, 'regenerate' otherwise")
    reasoning = dspy.OutputField(desc="Explanation of the evaluation")


class OpenEndedSuitabilitySignature(dspy.Signature):
    """Evaluates suitability for Open-Ended questions."""
    question = dspy.InputField(desc="The open-ended question")
    solicits_ideas_or_opinions_score = dspy.OutputField(desc="Score 1-5: solicit ideas/opinions")
    discourages_one_word_answers_score = dspy.OutputField(desc="Score 1-5: discourage one-word answers")
    child_cannot_opt_out_score = dspy.OutputField(desc="Score 1-5: difficulty to opt out")
    may_connect_to_personal_experience_score = dspy.OutputField(desc="Score 1-5: allows personal experiences")
    suitability_score = dspy.OutputField(desc="Average score 1-5 across four criteria")
    decision = dspy.OutputField(desc="'pass' if suitability_score >= 3, 'regenerate' otherwise")
    reasoning = dspy.OutputField(desc="Explanation of the evaluation")


class WhSuitabilitySignature(dspy.Signature):
    """Evaluates suitability for Wh- questions."""
    question = dspy.InputField(desc="The Wh- question")
    focuses_on_story_details = dspy.OutputField(desc="True/False: focuses on story details?")
    suitability_score = dspy.OutputField(desc="Score from 0-1 (1 if criteria met, 0 otherwise)")
    decision = dspy.OutputField(desc="'pass' if criteria met, 'regenerate' otherwise")
    reasoning = dspy.OutputField(desc="Explanation of the evaluation")


class DistancingSuitabilitySignature(dspy.Signature):
    """Evaluates suitability for Distancing questions."""
    question = dspy.InputField(desc="The distancing question")
    asks_about_child_experiences_score = dspy.OutputField(desc="Score 1-5: asks about personal experiences")
    cannot_answer_one_word_score = dspy.OutputField(desc="Score 1-5: cannot answer in one word")
    relates_to_current_page_score = dspy.OutputField(desc="Score 1-5: relates to current page")
    suitability_score = dspy.OutputField(desc="Average score 1-5 across three criteria")
    decision = dspy.OutputField(desc="'pass' if suitability_score >= 3, 'regenerate' otherwise")
    reasoning = dspy.OutputField(desc="Explanation of the evaluation")


@register_core_evaluator("completion_suitability")
class CompletionEvaluator(BaseEvaluator):
    def evaluate(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            agent = dspy.Predict(CompletionSuitabilitySignature)
            result = agent(question=question)
            score = float(getattr(result, "suitability_score", 0.0))
            decision = getattr(result, "decision", "regenerate")
            reasoning = getattr(result, "reasoning", "")
            return {"score": score, "decision": decision, "reasoning": reasoning}
        except Exception as exc:
            print(f"[WARN] completion_suitability error: {exc}")
            return {"score": 0.0, "decision": "error", "reasoning": str(exc)}


@register_core_evaluator("recall_suitability")
class RecallEvaluator(BaseEvaluator):
    def evaluate(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            agent = dspy.Predict(RecallSuitabilitySignature)
            result = agent(question=question)
            score = float(getattr(result, "suitability_score", 0.0))
            decision = getattr(result, "decision", "regenerate")
            reasoning = getattr(result, "reasoning", "")
            return {"score": score, "decision": decision, "reasoning": reasoning}
        except Exception as exc:
            print(f"[WARN] recall_suitability error: {exc}")
            return {"score": 0.0, "decision": "error", "reasoning": str(exc)}


@register_core_evaluator("open_ended_suitability")
class OpenEndedEvaluator(BaseEvaluator):
    def evaluate(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            agent = dspy.Predict(OpenEndedSuitabilitySignature)
            result = agent(question=question)
            score = float(getattr(result, "suitability_score", 0.0))
            # If model doesn't provide decision, derive it
            derived = "pass" if score >= 3.0 else "regenerate"
            decision = getattr(result, "decision", derived)
            reasoning = getattr(result, "reasoning", "")
            return {"score": score, "decision": decision, "reasoning": reasoning}
        except Exception as exc:
            print(f"[WARN] open_ended_suitability error: {exc}")
            return {"score": 0.0, "decision": "error", "reasoning": str(exc)}


@register_core_evaluator("wh_suitability")
class WhQuestionEvaluator(BaseEvaluator):
    def evaluate(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            agent = dspy.Predict(WhSuitabilitySignature)
            result = agent(question=question)
            score = float(getattr(result, "suitability_score", 0.0))
            decision = getattr(result, "decision", "regenerate")
            reasoning = getattr(result, "reasoning", "")
            return {"score": score, "decision": decision, "reasoning": reasoning}
        except Exception as exc:
            print(f"[WARN] wh_suitability error: {exc}")
            return {"score": 0.0, "decision": "error", "reasoning": str(exc)}


@register_core_evaluator("distancing_suitability")
class DistancingEvaluator(BaseEvaluator):
    def evaluate(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            agent = dspy.Predict(DistancingSuitabilitySignature)
            result = agent(question=question)
            score = float(getattr(result, "suitability_score", 0.0))
            derived = "pass" if score >= 3.0 else "regenerate"
            decision = getattr(result, "decision", derived)
            reasoning = getattr(result, "reasoning", "")
            return {"score": score, "decision": decision, "reasoning": reasoning}
        except Exception as exc:
            print(f"[WARN] distancing_suitability error: {exc}")
            return {"score": 0.0, "decision": "error", "reasoning": str(exc)}


class DynamicEvaluator(BaseEvaluator):
    def evaluate(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError("DynamicEvaluator.evaluate() not implemented yet.")


set_fallback_evaluator(DynamicEvaluator)
load_evaluator_metadata()


def list_registered_evaluators() -> Dict[str, Dict[str, Any]]:
    return {name: get_evaluator_metadata(name) for name in list_core_evaluators()}


def list_registered_dynamic_evaluators() -> Dict[str, Dict[str, Any]]:
    return {name: get_evaluator_metadata(name) for name in list_dynamic_evaluators()}


__all__ = [
    "BaseEvaluator",
    "CompletionEvaluator",
    "RecallEvaluator",
    "OpenEndedEvaluator",
    "WhQuestionEvaluator",
    "DistancingEvaluator",
    "DynamicEvaluator",
    "create_evaluator",
    "get_evaluator_metadata",
    "persist_dynamic_metadata",
    "list_registered_evaluators",
    "list_registered_dynamic_evaluators",
]

