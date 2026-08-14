"""Rich table and panel renderers for lineage read-back output."""

from __future__ import annotations

import json
from collections.abc import Sequence

from rich.panel import Panel
from rich.table import Table

from refinely.tracking.models import CaseRecord, CompileRecord, EvaluationRun


def _run_id(run_id: str) -> str:
    return run_id[:8]


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _metric_names(runs: Sequence[EvaluationRun]) -> list[str]:
    names: set[str] = set()
    for run in runs:
        names.update(run.metric_results)
    return sorted(names)


def runs_table(runs: Sequence[EvaluationRun]) -> Table:
    table = Table(title="Runs")
    table.add_column("run_id")
    table.add_column("created_at", overflow="fold")
    table.add_column("aggregate_score", justify="right")
    table.add_column("trial", justify="right")
    metrics = _metric_names(runs)
    for metric in metrics:
        table.add_column(metric, justify="right")
    for run in runs:
        table.add_row(
            _run_id(run.run_id),
            run.created_at,
            _fmt(run.aggregate_score),
            "" if run.optuna_trial_number is None else str(run.optuna_trial_number),
            *[_fmt(run.metric_results[m]) if m in run.metric_results else "n/a" for m in metrics],
        )
    return table


def cases_table(cases: Sequence[CaseRecord]) -> Table:
    table = Table(title="Cases (worst first)")
    table.add_column("case_id")
    table.add_column("score", justify="right")
    metrics = sorted({m for case in cases for m in case.metric_scores})
    for metric in metrics:
        table.add_column(metric, justify="right")
    table.add_column("input", overflow="fold")
    table.add_column("expected", overflow="fold")
    table.add_column("output", overflow="fold")
    table.add_column("error", overflow="fold")
    for case in cases:
        table.add_row(
            case.case_id,
            _fmt(case.score),
            *[_fmt(case.metric_scores[m]) if m in case.metric_scores else "n/a" for m in metrics],
            json.dumps(case.input, ensure_ascii=False),
            json.dumps(case.expected, ensure_ascii=False),
            json.dumps(case.output, ensure_ascii=False) if case.output is not None else "",
            case.error or "",
        )
    return table


def config_delta(current: dict, baseline: dict) -> dict[str, tuple[str, object, object]]:
    """Key-level diff of `current` vs `baseline` configs.

    Returns a dict mapping key → (change, before, after) where `change` is one of
    "added", "removed", or "changed". Keys with equal values are omitted.
    """
    delta: dict[str, tuple[str, object, object]] = {}
    all_keys = set(current) | set(baseline)
    for key in sorted(all_keys):
        if key not in baseline:
            delta[key] = ("added", None, current[key])
        elif key not in current:
            delta[key] = ("removed", baseline[key], None)
        elif current[key] != baseline[key]:
            delta[key] = ("changed", baseline[key], current[key])
    return delta


def case_pair_table(
    pairs: Sequence[tuple[str, float | None, float | None, float | None]],
    baseline_run: EvaluationRun | None = None,
    current_run: EvaluationRun | None = None,
) -> Table:
    """Per-case paired delta table. Each pair is (case_id, before, after, delta);
    before/after are None when a case is missing from one run."""
    table = Table(title="Per-case comparison")
    table.add_column("case_id")
    table.add_column("before", justify="right")
    table.add_column("after", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("direction")
    broke = fixed = unchanged = 0
    for case_id, before, after, delta in pairs:
        if before is not None and after is not None:
            if delta is not None and abs(delta) < 1e-9:
                direction, unchanged = "unchanged", unchanged + 1
            elif delta is not None and delta < 0:
                direction, broke = "broke", broke + 1
            elif delta is not None:
                direction, fixed = "fixed", fixed + 1
            else:
                direction = "unchanged"
            table.add_row(
                case_id,
                _fmt(before) if before is not None else "",
                _fmt(after) if after is not None else "",
                f"{delta:+.4f}" if delta is not None else "",
                direction,
            )
        else:
            table.add_row(
                case_id,
                _fmt(before) if before is not None else "",
                _fmt(after) if after is not None else "",
                "",
                "only in one run",
            )
    return table


def case_pair_summary(pairs: Sequence[tuple[str, float | None, float | None, float | None]]) -> str:
    broke = fixed = unchanged = only_one = 0
    for _, before, after, delta in pairs:
        if before is None or after is None:
            only_one += 1
        elif abs(delta) < 1e-9:
            unchanged += 1
        elif delta < 0:
            broke += 1
        else:
            fixed += 1
    return f"{broke} broke / {fixed} fixed / {unchanged} unchanged" + (
        f" (+{only_one} only in one run)" if only_one else ""
    )


def _metric_cell(run: EvaluationRun, metric: str, ref: EvaluationRun | None) -> str:
    value = run.metric_results.get(metric)
    if value is None:
        return "n/a"
    if ref is None:
        return _fmt(value)
    ref_value = ref.metric_results.get(metric)
    if ref_value is None:
        return _fmt(value)
    delta = value - ref_value
    if abs(delta) < 1e-9:
        return f"{_fmt(value)} (unchanged)"
    return f"{_fmt(value)} ({delta:+.4f})"


def compare_table(
    runs_chronological: Sequence[EvaluationRun],
    baseline_run: EvaluationRun | None = None,
) -> Table:
    table = Table(title="Compare")
    table.add_column("run_id")
    table.add_column("created_at", overflow="fold")
    table.add_column("model", justify="left")
    table.add_column("aggregate_score", justify="right")
    metrics = _metric_names(runs_chronological)
    for metric in metrics:
        table.add_column(metric, justify="right")

    for index, run in enumerate(runs_chronological):
        if baseline_run is None:
            is_baseline = index == 0
            ref = runs_chronological[index - 1] if index > 0 else None
        else:
            is_baseline = run.run_id == baseline_run.run_id
            ref = None if is_baseline else baseline_run
        run_id = _run_id(run.run_id)
        if is_baseline:
            run_id = f"{run_id} (baseline)"
        table.add_row(
            run_id,
            run.created_at,
            run.model_name or "",
            _fmt(run.aggregate_score),
            *[_metric_cell(run, metric, ref) for metric in metrics],
        )
    return table


def best_run_panel(best_run: EvaluationRun) -> Panel:
    return Panel(
        "\n".join(
            [
                f"run id: {best_run.run_id}",
                f"aggregate_score: {_fmt(best_run.aggregate_score)}",
                f"configuration: {best_run.configuration}",
            ]
        ),
        title="Best run",
    )


def best_compile_panel(best_compile: CompileRecord) -> Panel:
    return Panel(
        "\n".join(
            [
                f"compile id: {best_compile.compile_id}",
                f"compiled_score: {_fmt(best_compile.compiled_score)}",
            ]
        ),
        title="Best compile",
    )
