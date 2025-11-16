from __future__ import annotations

from typing import Dict, Iterable, Optional, Type

from .base_evaluator import BaseEvaluator


CORE_EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {}
DYNAMIC_EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {}
_FALLBACK_EVALUATOR_CLASS: Optional[Type[BaseEvaluator]] = None


def register_core_evaluator(name: str):
    """Decorator to register a core evaluator class."""

    def decorator(cls: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
        CORE_EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_dynamic_evaluator(name: str):
    """Decorator to register a dynamic evaluator class."""

    def decorator(cls: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
        DYNAMIC_EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorator


def set_fallback_evaluator(cls: Type[BaseEvaluator]) -> None:
    """Sets the fallback evaluator class used when a name is unregistered."""
    global _FALLBACK_EVALUATOR_CLASS
    _FALLBACK_EVALUATOR_CLASS = cls


def get_evaluator_class(name: str) -> Type[BaseEvaluator]:
    if name in CORE_EVALUATOR_REGISTRY:
        return CORE_EVALUATOR_REGISTRY[name]
    if name in DYNAMIC_EVALUATOR_REGISTRY:
        return DYNAMIC_EVALUATOR_REGISTRY[name]
    if _FALLBACK_EVALUATOR_CLASS:
        return _FALLBACK_EVALUATOR_CLASS
    raise ValueError(f"Evaluator '{name}' not found.")


def create_evaluator(name: str, **kwargs) -> BaseEvaluator:
    cls = get_evaluator_class(name)
    return cls(**kwargs)


def list_core_evaluators() -> Iterable[str]:
    return CORE_EVALUATOR_REGISTRY.keys()


def list_dynamic_evaluators() -> Iterable[str]:
    return DYNAMIC_EVALUATOR_REGISTRY.keys()


def clear_dynamic_registry() -> None:
    DYNAMIC_EVALUATOR_REGISTRY.clear()

