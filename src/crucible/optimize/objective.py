"""Optuna objective functions that wrap full evaluation runs."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna

from crucible.core.settings import Settings
from crucible.eval.datasets import EvalCase
from crucible.eval.metrics import Metric
from crucible.eval.runner import EvaluationRunner
from crucible.llm.client import LLMClient
from crucible.registry import get_registration
from crucible.tracking.db import LineageDB


def build_objective(
    *,
    app_name: str,
    app: Any,
    dataset: list[EvalCase],
    dataset_version: str,
    lineage_db_path: str | Path,
    client: LLMClient,
    settings: Settings | None = None,
    metrics: list[Metric] | None = None,
    search_space: Callable[[optuna.trial.Trial], dict[str, Any]] | None = None,
    weights: dict[str, float] | None = None,
) -> Callable[[optuna.trial.Trial], float]:
    """Build an Optuna objective that evaluates a sampled config and records it to lineage.

    The returned callable takes an Optuna trial; each call samples a
    configuration from the app's search space, runs a full `EvaluationRunner`
    pass, records the run to lineage with `optuna_trial_number` set, and
    returns the run's `aggregate_score` as the value to maximize.

    `metrics`, `search_space`, and `weights` default to the app's registered
    ones; pass them explicitly to run apps without registrations.
    """
    settings = settings or Settings()
    if weights is None:
        weights = get_registration(app_name).weights
    if metrics is None:
        metrics = get_registration(app_name).metrics_factory(client, settings)
    if search_space is None:
        search_space = get_registration(app_name).search_space
    runner = EvaluationRunner(metrics, app_name, weights=weights)
    db = LineageDB(lineage_db_path)
    db.init_schema()

    def objective(trial: optuna.trial.Trial) -> float:
        config = search_space(trial)
        result = runner.run(dataset, app, config=config, dataset_version=dataset_version)
        db.record_run(
            app_name=app_name,
            dataset_version=dataset_version,
            configuration=config,
            aggregate_score=result.aggregate_score,
            metric_results=result.metric_results,
            case_results=result.case_results,
            weights=weights,
            optuna_trial_number=trial.number,
        )
        return result.aggregate_score

    return objective
