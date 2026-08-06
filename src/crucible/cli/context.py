"""Shared runtime helpers for CLI commands.

Kept in one module so tests can monkeypatch a single seam (e.g.
``crucible.cli.context.get_registration``) and every command picks up the
override at call time.
"""

import click

from crucible.core.settings import Settings
from crucible.eval.datasets import EvalCase, dataset_version, load_dataset
from crucible.llm.client import AsyncOpenAIClient, LLMClient
from crucible.optimize.study import run_study
from crucible.registry import AppRegistration, get_registration, registered_apps
from crucible.tracking.db import LineageDB


def _client(settings: Settings) -> LLMClient:
    if not settings.has_api_key:
        raise click.ClickException(
            "CRUCIBLE_OPENAI_API_KEY is not set; set the env var or add a .env file"
        )
    return AsyncOpenAIClient(api_key=settings.openai_api_key, base_url=settings.base_url)


def _load_run_context(
    app: str,
) -> tuple[AppRegistration, Settings, LLMClient, list[EvalCase], str]:
    """Build the shared evaluation prelude: registration, settings, client, dataset, version."""
    registration = get_registration(app)
    settings = Settings()
    client = _client(settings)
    dataset = load_dataset(registration.dataset_path)
    version = dataset_version(registration.dataset_path)
    return registration, settings, client, dataset, version


def _resolve_run_id(db: LineageDB, app: str, value: str, label: str = "Run") -> str:
    """Resolve a full run id or a unique abbreviated prefix to a full run id."""
    if db.run_exists(value):
        return value
    matches = db.find_runs_by_prefix(app, value)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise click.ClickException(
            f"{label} prefix {value!r} is ambiguous for app {app!r} "
            f"(matches {len(matches)} runs); use a longer prefix."
        )
    raise click.ClickException(f"{label} {value!r} not found for app {app!r}")


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


__all__ = [
    "_client",
    "_format_counts",
    "_load_run_context",
    "_resolve_run_id",
    "get_registration",
    "registered_apps",
    "run_study",
]
