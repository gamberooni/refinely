import csv
import io
import json

from rich.console import Console

from refinely.reporting.export import export_runs_csv, export_runs_json
from refinely.reporting.render import (
    _metric_names,
    best_compile_panel,
    best_run_panel,
    cases_table,
    compare_table,
    runs_table,
)
from refinely.tracking.models import CaseRecord, CompileRecord, EvaluationRun


def _render(table) -> str:
    buffer = io.StringIO()
    Console(file=buffer, width=200).print(table)
    return buffer.getvalue()


def _run(
    run_id: str,
    aggregate_score: float,
    metric_results: dict[str, float],
    *, 
    created_at: str = "2026-01-01T00:00:00+00:00",
    trial: int | None = None,
    configuration: dict | None = None,
    model_name: str | None = None,
) -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id,
        app_name="extraction",
        dataset_version="extraction_v1",
        created_at=created_at,
        aggregate_score=aggregate_score,
        optuna_trial_number=trial,
        metric_results=metric_results,
        configuration=configuration if configuration is not None else {},
        model_name=model_name,
    )


def test_metric_names_sorted_union() -> None:
    runs = [
        _run("a", 0.5, {"cost": 0.9}),
        _run("b", 0.6, {"exact_match": 0.6, "latency": 1.0}),
    ]

    assert _metric_names(runs) == ["cost", "exact_match", "latency"]


def test_runs_table_renders_columns_and_missing_metric() -> None:
    runs = [
        _run("abc123456789", 0.5, {"exact_match": 0.5, "latency": 1.0}, trial=0),
        _run("def456789012", 0.7, {"exact_match": 0.7}),
    ]

    text = _render(runs_table(runs))

    assert "Runs" in text
    assert "run_id" in text
    assert "aggregate_score" in text
    assert "exact_match" in text
    assert "latency" in text
    assert "abc12345" in text
    assert "0.7000" in text
    assert text.count("-") >= 1


def test_cases_table_renders_metric_scores_and_none_output() -> None:
    cases = [
        CaseRecord(
            case_id="c0",
            score=0.7,
            metric_scores={"exact_match": 1.0, "latency": 1.0},
            input={"text": "in"},
            expected={"v": 1},
            output=None,
        ),
        CaseRecord(
            case_id="c1",
            score=0.15,
            metric_scores={"exact_match": 0.0, "cost": 1.0},
            input={"text": "in2"},
            expected={"v": 2},
            output={"v": 2},
        ),
    ]

    text = _render(cases_table(cases))

    assert "Cases (worst first)" in text
    assert "exact_match" in text
    assert "cost" in text
    assert "latency" in text
    assert "1.0000" in text
    assert "0.0000" in text
    assert '{"text": "in2"}' in text


def test_compare_table_default_prev_run_deltas() -> None:
    runs = [
        _run("a", 0.55, {"exact_match": 0.55}, created_at="2026-01-01"),
        _run("b", 0.8, {"exact_match": 0.8}, created_at="2026-01-02"),
        _run("c", 0.66, {"exact_match": 0.66}, created_at="2026-01-03"),
    ]

    text = _render(compare_table(runs))

    assert "a (baseline)" in text
    assert "0.8000 (+0.2500)" in text
    assert "0.6600 (-0.1400)" in text
    assert "0.5500" in text


def test_compare_table_with_baseline_run_shows_bare_baseline_values() -> None:
    baseline = _run("b", 0.5, {"exact_match": 0.5}, created_at="2026-01-02")
    runs = [
        _run("a", 0.6, {"exact_match": 0.6}, created_at="2026-01-01"),
        baseline,
        _run("c", 0.7, {"exact_match": 0.7}, created_at="2026-01-03"),
    ]

    text = _render(compare_table(runs, baseline_run=baseline))

    assert "b (baseline)" in text
    assert "0.6000 (+0.1000)" in text
    assert "0.7000 (+0.2000)" in text
    assert "0.5000" in text
    assert "(unchanged)" not in text


def test_compare_table_metric_missing_in_some_runs() -> None:
    baseline = _run("b", 0.5, {"exact_match": 0.5})
    runs = [
        _run("a", 0.5, {"exact_match": 0.5}),
        baseline,
        _run("c", 0.7, {"exact_match": 0.7, "latency": 1.0}),
    ]

    text = _render(compare_table(runs, baseline_run=baseline))

    assert "-" in text
    assert "1.0000" in text


def test_compare_table_renders_model_column() -> None:
    baseline = _run("b", 0.5, {"exact_match": 0.5}, model_name="gpt-4o")
    runs = [
        _run("a", 0.4, {"exact_match": 0.4}, model_name=None),
        baseline,
        _run("c", 0.7, {"exact_match": 0.7}, model_name="claude-3"),
    ]

    text = _render(compare_table(runs, baseline_run=baseline))

    assert "gpt-4o" in text
    assert "claude-3" in text


def test_best_run_and_compile_panels_render_content() -> None:
    run = _run(
        "abcdef0123456789",
        0.9,
        {"exact_match": 0.9},
        configuration={"temperature": 0.4},
    )
    compile_row = CompileRecord(
        compile_id="cdef0123456789",
        app_name="extraction",
        dataset_version="extraction_v1",
        optimizer="BootstrapFewShot",
        configuration={},
        artifact_path="optimized_program.json",
        baseline_score=0.0,
        compiled_score=0.88,
        created_at="2026-01-01T00:00:00+00:00",
    )

    run_text = _render(best_run_panel(run))
    compile_text = _render(best_compile_panel(compile_row))

    assert "Best run" in run_text
    assert "run id: abcdef0123456789" in run_text
    assert "aggregate_score: 0.9000" in run_text
    assert "Best compile" in compile_text
    assert "compile id: cdef0123456789" in compile_text
    assert "compiled_score: 0.8800" in compile_text


def test_export_runs_csv_includes_configuration_column(tmp_path) -> None:
    runs = [
        _run(
            "abc",
            0.5,
            {"exact_match": 0.5},
            trial=2,
            configuration={"temperature": 0.3},
        )
    ]
    path = tmp_path / "runs.csv"

    export_runs_csv(runs, path)

    rows = list(csv.reader(open(path, encoding="utf-8")))
    assert rows[0] == [
        "run_id",
        "created_at",
        "aggregate_score",
        "optuna_trial_number",
        "configuration",
        "exact_match",
    ]
    assert rows[1][4] == '{"temperature": 0.3}'
    assert rows[1][5] == "0.5"


def test_export_runs_csv_blanks_trial_and_missing_metric(tmp_path) -> None:
    runs = [
        _run("abc", 0.5, {"exact_match": 0.5}),
    ]
    path = tmp_path / "runs.csv"

    export_runs_csv(runs, path)

    rows = list(csv.reader(open(path, encoding="utf-8")))
    assert rows[1][3] == ""
    assert rows[1][5] == "0.5"


def test_export_runs_csv_empty_runs_writes_header_only(tmp_path) -> None:
    path = tmp_path / "empty.csv"

    export_runs_csv([], path)

    content = path.read_text()
    assert content.startswith("run_id,created_at,aggregate_score,optuna_trial_number")
    assert "configuration" in content


def test_export_runs_json_writes_list_with_configuration(tmp_path) -> None:
    runs = [
        _run(
            "abc",
            0.5,
            {"exact_match": 0.5},
            configuration={"temperature": 0.3},
        )
    ]
    path = tmp_path / "runs.json"

    export_runs_json(runs, path)

    data = json.loads(path.read_text())
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["run_id"] == "abc"
    assert data[0]["configuration"] == {"temperature": 0.3}
