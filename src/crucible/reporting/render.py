"""Rich table and panel renderers for lineage read-back output."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from rich.panel import Panel
from rich.table import Table

Run = dict[str, Any]


def _run_id(run_id: str) -> str:
    return run_id[:8]


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _metric_names(runs: Sequence[Run]) -> list[str]:
    names: set[str] = set()
    for run in runs:
        names.update(run.get("metric_results", {}))
    return sorted(names)


def runs_table(runs: Sequence[Run]) -> Table:
    table = Table(title="Runs")
    table.add_column("run_id")
    table.add_column("created_at", overflow="fold")
    table.add_column("aggregate_score", justify="right")
    table.add_column("trial", justify="right")
    metrics = _metric_names(runs)
    for metric in metrics:
        table.add_column(metric, justify="right")
    for run in runs:
        metric_values = run.get("metric_results", {})
        table.add_row(
            _run_id(run["run_id"]),
            run["created_at"],
            _fmt(run["aggregate_score"]),
            "" if run["optuna_trial_number"] is None else str(run["optuna_trial_number"]),
            *[_fmt(metric_values[m]) if m in metric_values else "-" for m in metrics],
        )
    return table


def cases_table(cases: Sequence[dict[str, Any]]) -> Table:
    table = Table(title="Cases (worst first)")
    table.add_column("case_id")
    table.add_column("score", justify="right")
    table.add_column("input", overflow="fold")
    table.add_column("expected", overflow="fold")
    table.add_column("output", overflow="fold")
    for case in cases:
        table.add_row(
            case["case_id"],
            _fmt(case["score"]),
            json.dumps(case["input"], ensure_ascii=False),
            json.dumps(case["expected"], ensure_ascii=False),
            json.dumps(case["output"], ensure_ascii=False) if case["output"] is not None else "",
        )
    return table


def _metric_cell(run: Run, metric: str, ref: Run | None) -> str:
    value = run.get("metric_results", {}).get(metric)
    if value is None:
        return "-"
    if ref is None:
        return _fmt(value)
    ref_value = ref.get("metric_results", {}).get(metric)
    if ref_value is None:
        return _fmt(value)
    delta = value - ref_value
    if abs(delta) < 1e-9:
        return f"{_fmt(value)} (unchanged)"
    return f"{_fmt(value)} ({delta:+.4f})"


def compare_table(
    runs_chronological: Sequence[Run],
    baseline_run: Run | None = None,
) -> Table:
    table = Table(title="Compare")
    table.add_column("run_id")
    table.add_column("created_at", overflow="fold")
    table.add_column("aggregate_score", justify="right")
    metrics = _metric_names(runs_chronological)
    for metric in metrics:
        table.add_column(metric, justify="right")

    for index, run in enumerate(runs_chronological):
        if baseline_run is None:
            is_baseline = index == 0
            ref = runs_chronological[index - 1] if index > 0 else None
        else:
            is_baseline = run["run_id"] == baseline_run["run_id"]
            ref = None if is_baseline else baseline_run
        run_id = _run_id(run["run_id"])
        if is_baseline:
            run_id = f"{run_id} (baseline)"
        table.add_row(
            run_id,
            run["created_at"],
            _fmt(run["aggregate_score"]),
            *[_metric_cell(run, metric, ref) for metric in metrics],
        )
    return table


def best_run_panel(best_run: Run) -> Panel:
    return Panel(
        "\n".join(
            [
                f"run id: {best_run['run_id']}",
                f"aggregate_score: {_fmt(best_run['aggregate_score'])}",
                f"configuration: {best_run['configuration']}",
            ]
        ),
        title="Best run",
    )


def best_compile_panel(best_compile: Run) -> Panel:
    return Panel(
        "\n".join(
            [
                f"compile id: {best_compile['compile_id']}",
                f"compiled_score: {_fmt(best_compile['compiled_score'])}",
            ]
        ),
        title="Best compile",
    )
