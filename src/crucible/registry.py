"""Application registry: apps register themselves; framework core stays agnostic."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from crucible.core.exceptions import EvalError
from crucible.core.settings import Settings
from crucible.eval.metrics import Metric
from crucible.llm.client import LLMClient

if TYPE_CHECKING:
    from crucible.dspy.spec import DspyProgramSpec


@dataclass(frozen=True)
class AppRegistration:
    name: str
    build_adapter: Callable[[LLMClient, Settings], Any]
    metrics_factory: Callable[[LLMClient, Settings], list[Metric]]
    search_space: Callable[[Any], dict[str, Any]]
    default_config: dict[str, Any]
    weights: dict[str, float]
    dataset_path: Path = field(default_factory=Path)
    dspy_factory: Callable[[Settings], DspyProgramSpec] | None = None


_REGISTRY: dict[str, AppRegistration] = {}


def register_app(registration: AppRegistration) -> None:
    if registration.name in _REGISTRY:
        raise EvalError(f"App already registered: {registration.name!r}")
    _REGISTRY[registration.name] = registration


def get_registration(app_name: str) -> AppRegistration:
    try:
        return _REGISTRY[app_name]
    except KeyError:
        raise EvalError(f"No app registered: {app_name!r}") from None


def registered_apps() -> list[str]:
    return sorted(_REGISTRY)


def discover_apps() -> list[str]:
    """Import every app declared as an entry point in group "crucible.apps".

    Entry point values are module paths; importing a module registers its
    apps via register_app. Returns the full sorted list of registered apps.
    """
    from importlib.metadata import entry_points

    for ep in entry_points(group="crucible.apps"):
        ep.load()
    return registered_apps()
