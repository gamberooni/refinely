from pathlib import Path

import pytest
from click.testing import CliRunner

from crucible.cli import _load_run_context, main
from crucible.core.settings import Settings
from crucible.llm.usage import Result, TokenUsage
from crucible.registry import AppRegistration
from crucible.tracking.db import LineageDB
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


def test_evaluate_program_gate_rejects_app_without_dspy_factory(
    tmp_path: Path,
    _hermetic_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crucible.cli._client", lambda settings: StubLLMClient())
    monkeypatch.setattr("crucible.cli.get_registration", lambda app: _registration())

    program = tmp_path / "program.json"
    program.write_text("{}")

    result = CliRunner().invoke(main, ["evaluate", "extraction", "--program", str(program)])

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

    result = CliRunner().invoke(main, ["evaluate", "extraction", "--program", str(program)])

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
