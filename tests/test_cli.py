import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from apps.extraction import EXTRACTION_WEIGHTS
from crucible.cli import _load_run_context, main
from crucible.core.settings import Settings
from crucible.eval.runner import CaseResult
from crucible.llm.usage import Result, TokenUsage
from crucible.registry import AppRegistration
from crucible.tracking.db import LineageDB, evaluation_runs_table
from tests.stub_llm import StubLLMClient

DATASET_PATH = Path("datasets/extraction_v1.json")


@pytest.fixture
def _hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def _registration(*, dspy_factory=None, build_adapter=None) -> AppRegistration:
    return AppRegistration(
        name="extraction",
        build_adapter=build_adapter or (lambda client, settings, program_path=None: object()),
        metrics_factory=lambda client, settings: [],
        search_space=lambda trial: {},
        default_config={},
        weights={},
        dataset_path=DATASET_PATH,
        dspy_factory=dspy_factory,
    )


class _StubApp:
    def execute(self, input: dict, config: dict) -> Result:
        return Result(
            output={"field_value": "positive"},
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
            latency_seconds=0.1,
        )


def _case_results(n: int = 3) -> list[CaseResult]:
    return [
        CaseResult(
            case_id=f"c{i}",
            input={"text": f"input {i}"},
            output={"field_value": "positive"},
            expected={"field_name": "sentiment", "field_value": "positive"},
            scores={"exact_match": 1.0 if i == 0 else 0.0, "latency": 1.0, "cost": 1.0},
        )
        for i in range(n)
    ]


def _seed_runs(db_path: Path, scores: list[float], *, cases: bool = False) -> list[str]:
    db = LineageDB(db_path)
    db.init_schema()
    run_ids: list[str] = []
    for i, score in enumerate(scores):
        run_id = db.record_run(
            app_name="extraction",
            dataset_version="extraction_v1",
            configuration={"temperature": 0.1 + i},
            aggregate_score=score,
            metric_results={"exact_match": score, "cost": 1.0, "latency": 1.0},
            case_results=_case_results() if cases else [],
            weights=EXTRACTION_WEIGHTS,
            optuna_trial_number=i,
        )
        run_ids.append(run_id)
    with db._engine.begin() as conn:
        for i, run_id in enumerate(run_ids):
            conn.execute(
                evaluation_runs_table.update()
                .where(evaluation_runs_table.c.run_id == run_id)
                .values(created_at=datetime(2026, 1, i + 1, tzinfo=UTC).isoformat())
            )
    db.close()
    return run_ids


def _invoke(args: list[str]) -> None:
    return CliRunner().invoke(main, args, env={"COLUMNS": "200"})


def test_evaluate_program_gate_rejects_app_without_dspy_factory(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crucible.cli._client", lambda settings: StubLLMClient())
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())

    program = tmp_path / "program.json"
    program.write_text("{}")

    result = _invoke(["evaluate", "extraction", "--program", str(program)])

    assert result.exit_code == 1
    assert "support" in result.output
    assert "extraction" in result.output


def test_evaluate_program_passes_program_path_to_build_adapter(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    received: dict[str, object] = {}

    def _build_adapter(client, settings, program_path=None):
        received["program_path"] = program_path
        return _StubApp()

    monkeypatch.setattr("crucible.cli._client", lambda settings: StubLLMClient())
    monkeypatch.setattr(
        "crucible.cli.get_registration",
        lambda app: _registration(
            dspy_factory=lambda settings: object(),
            build_adapter=_build_adapter,
        ),
    )

    program = tmp_path / "program.json"
    program.write_text("{}")

    result = _invoke(["evaluate", "extraction", "--program", str(program)])

    assert result.exit_code == 0, result.output
    assert received["program_path"] == str(program)


def test_load_run_context_returns_expected_shape(
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = StubLLMClient()
    monkeypatch.setattr("crucible.cli._client", lambda settings: client)
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())

    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        "crucible.cli.load_dataset",
        lambda path: calls.append(("load_dataset", path)) or [],
    )
    monkeypatch.setattr(
        "crucible.cli.dataset_version",
        lambda path: calls.append(("dataset_version", path)) or "v1",
    )

    registration, settings, client_out, dataset, version = _load_run_context("extraction")

    assert registration.dataset_path == DATASET_PATH
    assert isinstance(settings, Settings)
    assert client_out is client
    assert dataset == []
    assert version == "v1"
    assert ("load_dataset", DATASET_PATH) in calls
    assert ("dataset_version", DATASET_PATH) in calls


def test_lineage_db_context_manager_inits_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "lineage.db"

    with LineageDB(db_path) as db:
        assert db.count_runs() == 0

    import sqlite3

    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    conn.close()
    assert {"evaluation_runs", "metric_results", "case_results", "dspy_compiles"} <= tables


def test_show_renders_runs_newest_first(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.55, 0.8, 0.66])

    result = _invoke(["show", "extraction"])

    assert result.exit_code == 0, result.output
    assert "aggregate_score" in result.output
    assert "exact_match" in result.output
    assert "0.8000" in result.output
    assert "0.5500" in result.output
    assert result.output.index(run_ids[2][:8]) < result.output.index(run_ids[0][:8])
    assert "Best run" in result.output
    assert f"run id: {run_ids[1]}" in result.output


def test_show_without_runs_prints_message(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())

    result = _invoke(["show", "extraction"])

    assert result.exit_code == 0, result.output
    assert "No runs recorded for app 'extraction'" in result.output


def test_show_run_renders_cases_worst_first(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.8], cases=True)

    result = _invoke(["show", "extraction", "--run", run_ids[0]])

    assert result.exit_code == 0, result.output
    assert "c0" in result.output
    assert "c1" in result.output
    assert "c2" in result.output
    assert "Cases (worst first)" in result.output
    assert result.output.index("c1") < result.output.index("c0")


def test_show_run_unknown_run_id_errors(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())

    result = _invoke(["show", "extraction", "--run", "nonexistent"])

    assert result.exit_code == 1
    assert "Run 'nonexistent' not found for app 'extraction'" in result.output


def test_compare_renders_deltas_against_previous_run(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())
    _seed_runs(tmp_path / "lineage.db", [0.55, 0.8, 0.66])

    result = _invoke(["compare", "extraction"])

    assert result.exit_code == 0, result.output
    assert "(baseline)" in result.output
    assert "0.8000 (+0.2500)" in result.output
    assert "0.6600 (-0.1400)" in result.output
    assert "(unchanged)" in result.output


def test_compare_with_baseline_override(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.55, 0.8, 0.66])

    result = _invoke(["compare", "extraction", "--baseline", run_ids[1]])

    assert result.exit_code == 0, result.output
    assert f"{run_ids[1][:8]} (baseline)" in result.output
    assert "0.5500 (-0.2500)" in result.output
    assert "0.6600 (-0.1400)" in result.output


def test_compare_unknown_baseline_errors(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())
    _seed_runs(tmp_path / "lineage.db", [0.8])

    result = _invoke(["compare", "extraction", "--baseline", "nonexistent"])

    assert result.exit_code == 1
    assert "Baseline run 'nonexistent' not found for app 'extraction'" in result.output


def test_export_csv_writes_file_and_echoes_path(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.55, 0.8])
    out = tmp_path / "out.csv"

    result = _invoke(["export", "extraction", "--output", str(out)])

    assert result.exit_code == 0, result.output
    assert "Wrote 2 runs to" in result.output
    content = out.read_text()
    assert content.startswith("run_id,created_at,aggregate_score,optuna_trial_number")
    assert "exact_match" in content
    assert run_ids[0] in content


def test_export_json_writes_valid_file(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.55, 0.8, 0.66])
    out = tmp_path / "out.json"

    result = _invoke(["export", "extraction", "--format", "json", "--output", str(out)])

    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert isinstance(data, list) and len(data) == 3
    assert data[0]["run_id"] == run_ids[2]
    assert data[0]["aggregate_score"] == pytest.approx(0.66)


def test_export_defaults_to_app_name_in_cwd(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())
    monkeypatch.chdir(tmp_path)
    _seed_runs(tmp_path / "lineage.db", [0.8])

    result = _invoke(["export", "extraction"])

    assert result.exit_code == 0, result.output
    assert "Wrote 1 runs to extraction_runs.csv" in result.output
    assert (tmp_path / "extraction_runs.csv").exists()


def test_export_zero_runs_writes_header_only(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())
    out = tmp_path / "empty.csv"

    result = _invoke(["export", "extraction", "--output", str(out)])

    assert result.exit_code == 0, result.output
    assert "Wrote 0 runs to" in result.output
    content = out.read_text()
    assert content.startswith("run_id,created_at,aggregate_score,optuna_trial_number")


def test_export_rejects_invalid_format(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())

    result = _invoke(["export", "extraction", "--format", "yaml"])

    assert result.exit_code == 2
    assert "'yaml' is not one of 'csv', 'json'" in result.output
