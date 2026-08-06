"""Scaffolding for new apps.

`write_app` emits an `apps/<name>.py` module (a ``register_app`` skeleton
following the ``apps/extraction.py`` convention) and a matching dataset stub.
"""

from __future__ import annotations

import re
from pathlib import Path

from crucible.core.exceptions import EvalError

APPS_DIR = Path("apps")
DATASETS_DIR = Path("datasets")
RESERVED_NAMES = {"crucible"}

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_APP_TEMPLATE = '''"""{name} app: ..."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crucible.core.exceptions import EvalError
from crucible.core.settings import Settings
from crucible.eval.metrics import CostMetric, LatencyMetric, Metric
from crucible.llm.client import LLMClient
from crucible.llm.usage import Result
from crucible.registry import AppRegistration, register_app

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "{dataset_file}"

DEFAULT_CONFIG: dict[str, Any] = {{}}
WEIGHTS: dict[str, float] = {{"exact_match": 0.7, "latency": 0.15, "cost": 0.15}}


class {AppName}App:
    def __init__(self, client: LLMClient, settings: Settings | None = None) -> None:
        self._client = client
        self._settings = settings or Settings()

    def execute(self, input: dict[str, Any], config: dict[str, Any]) -> Result:
        # TODO: implement the app using self._client and self._settings.model_name
        raise EvalError("not implemented")


def sample_{name}_config(trial: Any) -> dict[str, Any]:
    # TODO: sample one config from the search space, e.g. trial.suggest_float(...)
    return {{}}


def _build_adapter(
    client: LLMClient, settings: Settings | None = None
) -> {AppName}App:
    return {AppName}App(client, settings)


def _metrics_factory(
    client: LLMClient, settings: Settings | None = None
) -> list[Metric]:
    return [LatencyMetric(), CostMetric()]


register_app(
    AppRegistration(
        name="{name}",
        build_adapter=_build_adapter,
        metrics_factory=_metrics_factory,
        search_space=sample_{name}_config,
        default_config=DEFAULT_CONFIG,
        weights=WEIGHTS,
        dataset_path=DATASET_PATH,
    )
)
'''

_DATASET_TEMPLATE = '''{{
  "version": "{name}_v1",
  "cases": []
}}
'''


class ScaffoldError(EvalError):
    """Raised when an app cannot be scaffolded."""


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ScaffoldError(
            f"Invalid app name {name!r}: must be a valid Python identifier "
            "(letters, digits, underscores; cannot start with a digit)."
        )
    if name in RESERVED_NAMES:
        raise ScaffoldError(f"App name {name!r} is reserved and cannot be used.")
    return name


def app_module_path(name: str) -> Path:
    """Return the expected path for the app's module (without validating the name)."""
    return APPS_DIR / f"{name}.py"


def dataset_stub_path(name: str) -> Path:
    """Return the expected path for the app's dataset stub."""
    return DATASETS_DIR / f"{name}_v1.json"


def write_app(name: str, dataset_path: Path | None = None) -> tuple[Path, Path | None]:
    """Write an app module (and optionally a dataset stub) for a new app.

    Returns ``(app_path, dataset_path_or_None)``. Refuses to overwrite existing
    files. When ``dataset_path`` is given, the module points at it and no stub
    is written.
    """
    name = _validate_name(name)
    app_path = app_module_path(name)
    if app_path.exists():
        raise ScaffoldError(f"App module already exists: {app_path}")
    if dataset_path is None and dataset_stub_path(name).exists():
        raise ScaffoldError(f"Dataset file already exists: {dataset_stub_path(name)}")

    app_path.parent.mkdir(parents=True, exist_ok=True)
    app_name = "".join(part.capitalize() for part in name.split("_"))
    dataset_file = (
        dataset_path.name if dataset_path is not None else dataset_stub_path(name).name
    )
    source = _APP_TEMPLATE.format(
        name=name,
        AppName=app_name,
        dataset_file=dataset_file,
    )
    app_path.write_text(source)

    stub_path: Path | None = None
    if dataset_path is None:
        stub_path = dataset_stub_path(name)
        stub_path.parent.mkdir(parents=True, exist_ok=True)
        stub_path.write_text(_DATASET_TEMPLATE.format(name=name))

    return app_path, stub_path
