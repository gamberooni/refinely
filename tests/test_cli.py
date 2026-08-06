import csv
import json
from datetime import UTC, datetime, timedelta
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


def _registration(*, dspy_factory=None, build_adapter=None, default_config=None) -> AppRegistration:
    return AppRegistration(
        name="extraction",
        build_adapter=build_adapter or (lambda client, settings, program_path=None: object()),
        metrics_factory=lambda client, settings: [],
        search_space=lambda trial: {},
        default_config=default_config if default_config is not None else {},
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


class _FakeUuid:
    def __init__(self, n: int) -> None:
        self.hex = f"abcd{n:028d}"


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
                .values(
                    created_at=(
                        datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i)
                    ).isoformat()
                )
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
    monkeypatch.setattr("crucible.cli.context._client", lambda settings: StubLLMClient())
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())

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

    monkeypatch.setattr("crucible.cli.context._client", lambda settings: StubLLMClient())
    monkeypatch.setattr(
        "crucible.cli.context.get_registration",
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
    monkeypatch.setattr("crucible.cli.context._client", lambda settings: client)
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())

    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        "crucible.cli.context.load_dataset",
        lambda path: calls.append(("load_dataset", path)) or [],
    )
    monkeypatch.setattr(
        "crucible.cli.context.dataset_version",
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
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
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
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())

    result = _invoke(["show", "extraction"])

    assert result.exit_code == 0, result.output
    assert "No runs recorded for app 'extraction'" in result.output


def test_show_run_renders_cases_worst_first(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
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
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())

    result = _invoke(["show", "extraction", "--run", "nonexistent"])

    assert result.exit_code == 1
    assert "Run 'nonexistent' not found for app 'extraction'" in result.output


def test_compare_renders_deltas_against_previous_run(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
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
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
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
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    _seed_runs(tmp_path / "lineage.db", [0.8])

    result = _invoke(["compare", "extraction", "--baseline", "nonexistent"])

    assert result.exit_code == 1
    assert "Baseline run 'nonexistent' not found for app 'extraction'" in result.output


def test_show_run_finds_id_beyond_default_limit(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.1] * 55)

    result = _invoke(["show", "extraction", "--run", run_ids[0]])

    assert result.exit_code == 0, result.output
    assert "Cases (worst first)" in result.output


def test_show_pagination(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.1 + 0.01 * i for i in range(60)])

    result = _invoke(["show", "extraction", "--page", "2", "--limit", "10"])

    assert result.exit_code == 0, result.output
    assert "page 2 of 6" in result.output
    assert run_ids[45][:8] in result.output
    assert run_ids[40][:8] in result.output
    assert run_ids[30][:8] not in result.output


def test_show_page_out_of_range_errors(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    _seed_runs(tmp_path / "lineage.db", [0.1] * 5)

    result = _invoke(["show", "extraction", "--page", "2"])

    assert result.exit_code == 1
    assert "Page 2 is out of range for app 'extraction' (only 5 runs)" in result.output


def test_compare_pagination(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.1 + 0.01 * i for i in range(60)])

    result = _invoke(["compare", "extraction", "--page", "2", "--page-size", "10"])

    assert result.exit_code == 0, result.output
    assert "page 2 of 6" in result.output
    assert run_ids[10][:8] in result.output
    assert run_ids[19][:8] in result.output
    assert run_ids[9][:8] not in result.output


def test_compare_baseline_across_pages(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.1 + 0.05 * i for i in range(20)])

    result = _invoke(
        [
            "compare",
            "extraction",
            "--baseline",
            run_ids[0],
            "--page",
            "2",
            "--page-size",
            "10",
        ]
    )

    assert result.exit_code == 0, result.output
    assert "(baseline)" not in result.output
    assert "0.6000 (+0.5000)" in result.output
    assert "0.6500 (+0.5500)" in result.output


def test_compare_without_runs_prints_message(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())

    result = _invoke(["compare", "extraction"])

    assert result.exit_code == 0, result.output
    assert "No runs recorded for app 'extraction'" in result.output


def test_show_pager_pipes_all_runs(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.1 + 0.01 * i for i in range(60)])

    result = _invoke(["show", "extraction", "--pager"])

    assert result.exit_code == 0, result.output
    assert run_ids[0][:8] in result.output
    assert run_ids[59][:8] in result.output
    assert "page 2 of 6" not in result.output


def test_compare_pager_pipes_all_runs(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.1 + 0.01 * i for i in range(60)])

    result = _invoke(["compare", "extraction", "--pager"])

    assert result.exit_code == 0, result.output
    assert run_ids[0][:8] in result.output
    assert run_ids[59][:8] in result.output
    assert "(baseline)" in result.output
    assert "page 2 of 6" not in result.output


def test_export_csv_writes_file_and_echoes_path(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
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
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
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
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
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
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
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
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())

    result = _invoke(["export", "extraction", "--format", "yaml"])

    assert result.exit_code == 2
    assert "'yaml' is not one of 'csv', 'json'" in result.output


def test_show_run_accepts_abbreviated_prefix(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.8], cases=True)

    result = _invoke(["show", "extraction", "--run", run_ids[0][:8]])

    assert result.exit_code == 0, result.output
    assert "Cases (worst first)" in result.output


def test_show_run_ambiguous_prefix_errors(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    counter = {"n": 0}

    def _next_uuid() -> _FakeUuid:
        counter["n"] += 1
        return _FakeUuid(counter["n"])

    monkeypatch.setattr("crucible.tracking.db.uuid.uuid4", _next_uuid)
    _seed_runs(tmp_path / "lineage.db", [0.5, 0.6])

    result = _invoke(["show", "extraction", "--run", "abcd"])

    assert result.exit_code == 1
    assert "Run prefix 'abcd' is ambiguous for app 'extraction'" in result.output
    assert "use a longer prefix" in result.output


def test_compare_baseline_accepts_abbreviated_prefix(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.55, 0.8, 0.66])

    result = _invoke(["compare", "extraction", "--baseline", run_ids[1][:8]])

    assert result.exit_code == 0, result.output
    assert f"{run_ids[1][:8]} (baseline)" in result.output
    assert "0.5500 (-0.2500)" in result.output


def test_evaluate_config_merges_over_default_config(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context._client", lambda settings: StubLLMClient())
    monkeypatch.setattr(
        "crucible.cli.context.get_registration",
        lambda app: _registration(
            build_adapter=lambda client, settings, program_path=None: _StubApp(),
            default_config={"temperature": 0.1, "system_prompt_variant": "strict"},
        ),
    )

    result = _invoke(["evaluate", "extraction", "--config", '{"temperature": 0.9}'])

    assert result.exit_code == 0, result.output
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    runs = db.list_runs("extraction")
    assert runs[0].configuration == {
        "temperature": 0.9,
        "system_prompt_variant": "strict",
    }
    db.close()


def test_evaluate_config_without_override_uses_default(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context._client", lambda settings: StubLLMClient())
    monkeypatch.setattr(
        "crucible.cli.context.get_registration",
        lambda app: _registration(
            build_adapter=lambda client, settings, program_path=None: _StubApp(),
            default_config={"temperature": 0.1},
        ),
    )

    result = _invoke(["evaluate", "extraction"])

    assert result.exit_code == 0, result.output
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    assert db.list_runs("extraction")[0].configuration == {"temperature": 0.1}
    db.close()


def test_evaluate_config_rejects_invalid_json(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context._client", lambda settings: StubLLMClient())
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())

    result = _invoke(["evaluate", "extraction", "--config", "{not json"])

    assert result.exit_code == 1
    assert "Invalid --config JSON" in result.output


def test_evaluate_config_rejects_non_object(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context._client", lambda settings: StubLLMClient())
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())

    result = _invoke(["evaluate", "extraction", "--config", "[1, 2]"])

    assert result.exit_code == 1
    assert "--config must be a JSON object" in result.output


def test_export_csv_includes_configuration_column(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    _seed_runs(tmp_path / "lineage.db", [0.8])
    out = tmp_path / "out.csv"

    result = _invoke(["export", "extraction", "--output", str(out)])

    assert result.exit_code == 0, result.output
    rows = list(csv.reader(out.open()))
    assert "configuration" in rows[0]
    assert rows[1][rows[0].index("configuration")] == '{"temperature": 0.1}'


def _seed_tagged_runs(db_path: Path) -> list[str]:
    db = LineageDB(db_path)
    db.init_schema()
    run_ids: list[str] = []
    for i, (score, tags) in enumerate(
        [(0.8, ["candidate", "prod"]), (0.6, ["prod"]), (0.9, ["candidate"])]
    ):
        run_id = db.record_run(
            app_name="extraction",
            dataset_version="extraction_v1",
            configuration={"temperature": 0.1 + i},
            aggregate_score=score,
            metric_results={"exact_match": score, "cost": 1.0, "latency": 1.0},
            case_results=[],
            weights=EXTRACTION_WEIGHTS,
            tags=tags,
        )
        run_ids.append(run_id)
    with db._engine.begin() as conn:
        for i, run_id in enumerate(run_ids):
            conn.execute(
                evaluation_runs_table.update()
                .where(evaluation_runs_table.c.run_id == run_id)
                .values(
                    created_at=(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i)).isoformat()
                )
            )
    db.close()
    return run_ids


def test_evaluate_records_tags(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context._client", lambda settings: StubLLMClient())
    monkeypatch.setattr(
        "crucible.cli.context.get_registration",
        lambda app: _registration(
            build_adapter=lambda client, settings, program_path=None: _StubApp()
        ),
    )

    result = _invoke(["evaluate", "extraction", "--tags", "candidate,prod"])

    assert result.exit_code == 0, result.output
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    assert db.list_runs("extraction")[0].tags == "candidate,prod"
    db.close()


def test_show_filters_runs_by_tag(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_ids = _seed_tagged_runs(tmp_path / "lineage.db")

    result = _invoke(["show", "extraction", "--tag", "candidate"])

    assert result.exit_code == 0, result.output
    assert run_ids[0][:8] in result.output
    assert run_ids[2][:8] in result.output
    assert run_ids[1][:8] not in result.output


def test_show_tag_no_match_prints_message(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    _seed_runs(tmp_path / "lineage.db", [0.8])

    result = _invoke(["show", "extraction", "--tag", "nope"])

    assert result.exit_code == 0, result.output
    assert "no runs found matching the tag" in result.output


def test_compare_filters_by_tag(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_ids = _seed_tagged_runs(tmp_path / "lineage.db")

    result = _invoke(["compare", "extraction", "--tag", "candidate"])

    assert result.exit_code == 0, result.output
    assert run_ids[2][:8] in result.output
    assert run_ids[1][:8] not in result.output


def test_compare_tag_no_match_prints_message(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    _seed_runs(tmp_path / "lineage.db", [0.8])

    result = _invoke(["compare", "extraction", "--tag", "nope"])

    assert result.exit_code == 0, result.output
    assert "no runs found matching the tag" in result.output


def test_export_filters_by_tag(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_ids = _seed_tagged_runs(tmp_path / "lineage.db")
    out = tmp_path / "tagged.csv"

    result = _invoke(["export", "extraction", "--tag", "prod", "--output", str(out)])

    assert result.exit_code == 0, result.output
    content = out.read_text()
    assert run_ids[0] in content
    assert run_ids[1] in content
    assert run_ids[2] not in content


def test_compare_diff_config_shows_key_delta(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    _seed_runs(tmp_path / "lineage.db", [0.55, 0.8])

    result = _invoke(["compare", "extraction", "--diff-config"])

    assert result.exit_code == 0, result.output
    assert "Config delta vs baseline" in result.output
    assert "temperature" in result.output
    assert "changed" in result.output


def test_compare_diff_config_identical_configs(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    run_ids: list[str] = []
    for score in [0.55, 0.8]:
        run_ids.append(
            db.record_run(
                app_name="extraction",
                dataset_version="extraction_v1",
                configuration={"temperature": 0.1},
                aggregate_score=score,
                metric_results={"exact_match": score},
                case_results=[],
                weights=EXTRACTION_WEIGHTS,
            )
        )
    db.close()

    result = _invoke(
        ["compare", "extraction", "--diff-config", "--baseline", run_ids[0][:8]]
    )

    assert result.exit_code == 0, result.output
    assert "configurations are identical (no changes)" in result.output


def test_compare_cases_shows_broke_fixed_summary(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()

    def _scored(exact0: float, exact1: float, exact2: float) -> list[CaseResult]:
        return [
            CaseResult(
                case_id=f"c{i}",
                input={"text": f"input {i}"},
                output={"field_value": "positive"},
                expected={"field_name": "sentiment", "field_value": "positive"},
                scores={"exact_match": exact, "latency": 1.0, "cost": 1.0},
            )
            for i, exact in enumerate([exact0, exact1, exact2])
        ]

    run_ids: list[str] = []
    run_ids.append(
        db.record_run(
            app_name="extraction",
            dataset_version="extraction_v1",
            configuration={"temperature": 0.1},
            aggregate_score=0.8,
            metric_results={"exact_match": 0.8},
            case_results=_scored(1.0, 0.0, 0.5),
            weights=EXTRACTION_WEIGHTS,
        )
    )
    run_ids.append(
        db.record_run(
            app_name="extraction",
            dataset_version="extraction_v1",
            configuration={"temperature": 0.2},
            aggregate_score=0.6,
            metric_results={"exact_match": 0.6},
            case_results=_scored(0.0, 1.0, 0.5),
            weights=EXTRACTION_WEIGHTS,
        )
    )
    with db._engine.begin() as conn:
        for i, run_id in enumerate(run_ids):
            conn.execute(
                evaluation_runs_table.update()
                .where(evaluation_runs_table.c.run_id == run_id)
                .values(
                    created_at=(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i)).isoformat()
                )
            )
    db.close()

    result = _invoke(["compare", "extraction", "--cases"])

    assert result.exit_code == 0, result.output
    assert "broke" in result.output
    assert "fixed" in result.output


def test_compare_cases_needs_two_runs(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    _seed_runs(tmp_path / "lineage.db", [0.8], cases=True)

    result = _invoke(["compare", "extraction", "--cases"])

    assert result.exit_code == 0, result.output
    assert "comparison needs at least two matching runs" in result.output


def test_compare_cases_dataset_version_warning(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()

    def _cases() -> list[CaseResult]:
        return [
            CaseResult(
                case_id=f"c{i}",
                input={"text": f"input {i}"},
                output={"field_value": "positive"},
                expected={"field_name": "sentiment", "field_value": "positive"},
                scores={"exact_match": 1.0, "latency": 1.0, "cost": 1.0},
            )
            for i in range(2)
        ]

    db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration={},
        aggregate_score=0.8,
        metric_results={"exact_match": 0.8},
        case_results=_cases(),
        weights=EXTRACTION_WEIGHTS,
    )
    db.record_run(
        app_name="extraction",
        dataset_version="extraction_v2",
        configuration={},
        aggregate_score=0.6,
        metric_results={"exact_match": 0.6},
        case_results=_cases(),
        weights=EXTRACTION_WEIGHTS,
    )
    db.close()

    result = _invoke(["compare", "extraction", "--cases"])

    assert result.exit_code == 0, result.output
    assert "warning: dataset versions differ" in result.output


def _seed_run_with_errors(db_path: Path) -> str:
    db = LineageDB(db_path)
    db.init_schema()
    cases = _case_results(3)
    cases[1].error = "boom"
    run_id = db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration={},
        aggregate_score=0.5,
        metric_results={"exact_match": 0.5},
        case_results=cases,
        weights=EXTRACTION_WEIGHTS,
    )
    db.close()
    return run_id


def test_show_run_renders_error_column_and_count(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_id = _seed_run_with_errors(tmp_path / "lineage.db")

    result = _invoke(["show", "extraction", "--run", run_id[:8]])

    assert result.exit_code == 0, result.output
    assert "1 cases errored" in result.output
    assert "boom" in result.output


def test_show_run_clean_run_no_errored_count(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("crucible.cli.context.get_registration", lambda app: _registration())
    run_ids = _seed_runs(tmp_path / "lineage.db", [0.8], cases=True)

    result = _invoke(["show", "extraction", "--run", run_ids[0][:8]])

    assert result.exit_code == 0, result.output
    assert "cases errored" not in result.output
