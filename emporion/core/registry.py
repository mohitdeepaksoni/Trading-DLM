"""The plug-in registry — inherited verbatim in spirit from Agora.

Every extensible component (market loader, feature, baseline, evaluator, model,
broker adapter) registers under a *kind* and a *name*; configs refer to it by
name; the core never changes when the system grows.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

_REGISTRIES: Dict[str, Dict[str, Any]] = {}

KINDS = ("loader", "feature", "baseline", "evaluator", "model", "broker", "cost_model")


def register(kind: str, name: str) -> Callable[[Any], Any]:
    def _decorator(obj: Any) -> Any:
        _REGISTRIES.setdefault(kind, {})[name] = obj
        return obj
    return _decorator


def get(kind: str, name: str) -> Any:
    try:
        return _REGISTRIES[kind][name]
    except KeyError:
        raise KeyError(f"No {kind!r} registered as {name!r}. Available {kind!r}: {available(kind)}")


def available(kind: str) -> List[str]:
    return sorted(_REGISTRIES.get(kind, {}).keys())


def all_registered() -> Dict[str, List[str]]:
    return {k: available(k) for k in sorted(_REGISTRIES)}
