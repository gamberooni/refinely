"""CSV and JSON writers for lineage run export."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from refinely.tracking.models import EvaluationRun


def _metric_names(runs: Sequence[EvaluationRun]) -> list[str]:
    names: set[str] = set()
    for run in runs:
        names.update(run.metric_results)
    return sorted(names)


def export_runs_csv(runs: Sequence[EvaluationRun], path: str | Path) -> None:
    metrics = _metric_names(runs)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "run_id",
                "created_at",
                "aggregate_score",
                "optuna_trial_number",
                "configuration",
                *metrics,
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run.run_id,
                    run.created_at,
                    run.aggregate_score,
                    run.optuna_trial_number if run.optuna_trial_number is not None else "",
                    json.dumps(run.configuration, sort_keys=True),
                    *[run.metric_results.get(m, "") for m in metrics],
                ]
            )


def export_runs_json(runs: Sequence[EvaluationRun], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([run.model_dump() for run in runs], fh, indent=2)
