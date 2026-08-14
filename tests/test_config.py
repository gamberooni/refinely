import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner

from refinely.cli import main
from refinely.config import (
    ConfigError,
    clear_default,
    config_path,
    default_config,
    get_default,
    list_configs,
    rm_config,
    save_config,
    set_default,
    show_config,
    write_best_config,
)
from refinely.core.settings import Settings
from refinely.registry import AppRegistration
from refinely.tracking.db import LineageDB
from tests.stub_llm import StubLLMClient


@pytest.fixture
def _hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture
def _config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    target = tmp_path / "configs"
    monkeypatch.setattr("refinely.config.CONFIG_DIR", target)
    return target


class _StubApp:
    def execute(self, input: dict, config: dict) -> object:
        from refinely.llm.usage import Result, TokenUsage

        return Result(
            output={"field_value": "positive"},
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
            latency_seconds=0.1,
        )


def _registration(*, default_config=None, build_adapter=None) -> AppRegistration:
    return AppRegistration(
        name="extraction",
        build_adapter=build_adapter or (lambda client, settings, program_path=None: _StubApp()),
        metrics_factory=lambda client, settings: [],
        search_space=lambda trial: {},
        default_config=default_config if default_config is not None else {},
        weights={},
        dataset_path=Path("datasets/extraction_v1.json"),
    )


def _invoke(args: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    monkeypatch.setenv("REFINELY_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
    monkeypatch.setattr("refinely.cli.context._client", lambda settings: StubLLMClient())
    monkeypatch.setattr("refinely.cli.context.get_registration", lambda app: _registration())
    return CliRunner().invoke(main, args, env={"COLUMNS": "200"})


class TestConfigStorage:
    def test_save_and_show_roundtrip(self, _config_dir: Path) -> None:
        path = save_config("extraction", "my-run", {"temperature": 0.4})
        assert path == _config_dir / "extraction" / "my-run.json"
        assert path.exists()
        assert show_config("extraction", "my-run") == {"temperature": 0.4}

    def test_save_creates_parent_dirs(self, _config_dir: Path) -> None:
        path = save_config("rag", "nested", {"top_k": 3})
        assert path.exists()

    def test_save_rejects_invalid_names(self, _config_dir: Path) -> None:
        for bad in ("../escape", "a/b", ".hidden", "with space", "", "opt-best"):
            with pytest.raises(ConfigError):
                save_config("extraction", bad, {})

    def test_show_missing_raises(self, _config_dir: Path) -> None:
        with pytest.raises(ConfigError):
            show_config("extraction", "nope")

    def test_rm_deletes_and_clears_default(self, _config_dir: Path) -> None:
        save_config("extraction", "my-run", {"temperature": 0.4})
        set_default("extraction", "my-run")
        rm_config("extraction", "my-run")
        assert not (config_path("extraction", "my-run")).exists()
        assert get_default("extraction") is None

    def test_rm_missing_raises(self, _config_dir: Path) -> None:
        with pytest.raises(ConfigError):
            rm_config("extraction", "nope")

    def test_list_configs_groups_by_app(self, _config_dir: Path) -> None:
        save_config("extraction", "a", {})
        save_config("extraction", "b", {})
        save_config("rag", "c", {})
        assert list_configs() == {"extraction": ["a", "b"], "rag": ["c"]}
        assert list_configs("extraction") == {"extraction": ["a", "b"]}

    def test_list_configs_empty(self, _config_dir: Path) -> None:
        assert list_configs() == {}

    def test_default_pointer_roundtrip(self, _config_dir: Path) -> None:
        save_config("extraction", "my-run", {"temperature": 0.4})
        assert get_default("extraction") is None
        set_default("extraction", "my-run")
        assert get_default("extraction") == "my-run"
        clear_default("extraction")
        assert get_default("extraction") is None

    def test_default_config_no_pointer_returns_registered(self, _config_dir: Path) -> None:
        assert default_config("extraction", {"temperature": 0.0}) == {"temperature": 0.0}

    def test_default_config_pointer_merges_over_registered(self, _config_dir: Path) -> None:
        save_config("extraction", "my-run", {"temperature": 0.9})
        set_default("extraction", "my-run")
        merged = default_config("extraction", {"temperature": 0.0, "mode": "fast"})
        assert merged == {"temperature": 0.9, "mode": "fast"}

    def test_write_best_config_overwrites(self, _config_dir: Path) -> None:
        first = write_best_config("extraction", {"temperature": 0.1})
        assert first == _config_dir / "extraction" / "opt-best.json"
        write_best_config("extraction", {"temperature": 0.2})
        assert show_config("extraction", "opt-best") == {"temperature": 0.2}


class TestConfigCli:
    def test_save_reports_path(
        self, _config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = _invoke(
            ["config", "save", "my-run", "--app", "extraction", "--config", '{"temperature": 0.4}'],
            monkeypatch,
            tmp_path,
        )
        assert result.exit_code == 0, result.output
        assert str(_config_dir / "extraction" / "my-run.json") in result.output
        assert show_config("extraction", "my-run") == {"temperature": 0.4}

    def test_save_invalid_json_rejects_no_file(
        self, _config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = _invoke(
            ["config", "save", "my-run", "--app", "extraction", "--config", "{oops"],
            monkeypatch,
            tmp_path,
        )
        assert result.exit_code == 1
        assert "Invalid --config JSON" in result.output
        assert not (_config_dir / "extraction" / "my-run.json").exists()

    def test_save_non_object_rejects_no_file(
        self, _config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = _invoke(
            ["config", "save", "my-run", "--app", "extraction", "--config", "[1, 2]"],
            monkeypatch,
            tmp_path,
        )
        assert result.exit_code == 1
        assert "--config must be a JSON object" in result.output
        assert not (_config_dir / "extraction" / "my-run.json").exists()

    def test_list_marks_default_with_star(
        self, _config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        save_config("extraction", "a", {})
        save_config("extraction", "b", {})
        set_default("extraction", "a")
        result = _invoke(["config", "list"], monkeypatch, tmp_path)
        assert result.exit_code == 0, result.output
        assert "* extraction/a.json" in result.output
        assert " extraction/b.json" in result.output
        assert "default: a" in result.output

    def test_show_prints_json_contents(
        self, _config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        save_config("extraction", "my-run", {"temperature": 0.4})
        result = _invoke(["config", "show", "my-run", "--app", "extraction"], monkeypatch, tmp_path)
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"temperature": 0.4}

    def test_rm_deletes_file(
        self, _config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        save_config("extraction", "my-run", {})
        result = _invoke(["config", "rm", "my-run", "--app", "extraction"], monkeypatch, tmp_path)
        assert result.exit_code == 0, result.output
        assert not (config_path("extraction", "my-run")).exists()

    def test_default_set_and_clear(
        self, _config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        save_config("extraction", "my-run", {})
        result = _invoke(
            ["config", "default", "extraction", "--set", "my-run"], monkeypatch, tmp_path
        )
        assert result.exit_code == 0, result.output
        assert get_default("extraction") == "my-run"
        result = _invoke(["config", "default", "extraction", "--clear"], monkeypatch, tmp_path)
        assert result.exit_code == 0, result.output
        assert get_default("extraction") is None


class TestConfigResolution:
    def test_evaluate_accepts_config_name(
        self, _config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        save_config("extraction", "my-run", {"temperature": 0.4})
        monkeypatch.setattr(
            "refinely.cli.context.get_registration",
            lambda app: _registration(default_config={"temperature": 0.0}),
        )
        result = _invoke(["evaluate", "extraction", "--config", "my-run"], monkeypatch, tmp_path)
        assert result.exit_code == 0, result.output
        db = LineageDB(tmp_path / "lineage.db")
        db.init_schema()
        assert db.list_runs("extraction")[0].configuration == {"temperature": 0.4}
        db.close()

    def test_evaluate_unknown_config_name_errors(
        self, _config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = _invoke(["evaluate", "extraction", "--config", "nope"], monkeypatch, tmp_path)
        assert result.exit_code == 1
        assert "Config 'nope' not found for app 'extraction'" in result.output

    def test_evaluate_invalid_json_still_errors(
        self, _config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = _invoke(["evaluate", "extraction", "--config", "{not json"], monkeypatch, tmp_path)
        assert result.exit_code == 1
        assert "Invalid --config JSON" in result.output

    def test_evaluate_no_config_uses_default_pointer(
        self, _config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        save_config("extraction", "prod", {"temperature": 0.9})
        set_default("extraction", "prod")
        monkeypatch.setattr(
            "refinely.cli.context.get_registration",
            lambda app: _registration(default_config={"temperature": 0.0}),
        )
        result = _invoke(["evaluate", "extraction"], monkeypatch, tmp_path)
        assert result.exit_code == 0, result.output
        db = LineageDB(tmp_path / "lineage.db")
        db.init_schema()
        assert db.list_runs("extraction")[0].configuration == {"temperature": 0.9}
        db.close()


class TestModelAxis:
    def test_evaluate_records_model_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("REFINELY_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
        monkeypatch.setattr("refinely.cli.context._client", lambda settings: StubLLMClient())
        monkeypatch.setattr("refinely.cli.context.get_registration", lambda app: _registration())
        result = _invoke(["evaluate", "extraction", "--model", "gpt-4o"], monkeypatch, tmp_path)
        assert result.exit_code == 0, result.output
        db = LineageDB(tmp_path / "lineage.db")
        db.init_schema()
        assert db.list_runs("extraction")[0].model_name == "gpt-4o"
        db.close()

    def test_evaluate_records_default_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("REFINELY_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
        monkeypatch.setenv("REFINELY_MODEL_NAME", "default-model")
        monkeypatch.setattr("refinely.cli.context._client", lambda settings: StubLLMClient())
        monkeypatch.setattr("refinely.cli.context.get_registration", lambda app: _registration())
        result = _invoke(["evaluate", "extraction"], monkeypatch, tmp_path)
        assert result.exit_code == 0, result.output
        db = LineageDB(tmp_path / "lineage.db")
        db.init_schema()
        assert db.list_runs("extraction")[0].model_name == "default-model"
        db.close()

    def test_evaluate_models_fanout_records_each(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("REFINELY_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
        monkeypatch.setattr("refinely.cli.context._client", lambda settings: StubLLMClient())
        monkeypatch.setattr("refinely.cli.context.get_registration", lambda app: _registration())
        result = _invoke(
            ["evaluate", "extraction", "--models", "gpt-4o,claude-3"], monkeypatch, tmp_path
        )
        assert result.exit_code == 0, result.output
        db = LineageDB(tmp_path / "lineage.db")
        db.init_schema()
        models = sorted(r.model_name for r in db.list_runs("extraction"))
        assert models == ["claude-3", "gpt-4o"]
        db.close()

    def test_evaluate_models_empty_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("REFINELY_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
        monkeypatch.setattr("refinely.cli.context._client", lambda settings: StubLLMClient())
        monkeypatch.setattr("refinely.cli.context.get_registration", lambda app: _registration())
        result = _invoke(["evaluate", "extraction", "--models", ""], monkeypatch, tmp_path)
        assert result.exit_code == 1
        assert "--models must be a non-empty comma-separated list" in result.output

    def test_evaluate_model_and_models_mutually_exclusive(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("REFINELY_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
        monkeypatch.setattr("refinely.cli.context._client", lambda settings: StubLLMClient())
        monkeypatch.setattr("refinely.cli.context.get_registration", lambda app: _registration())
        result = _invoke(
            ["evaluate", "extraction", "--model", "gpt-4o", "--models", "gpt-4o,claude-3"],
            monkeypatch,
            tmp_path,
        )
        assert result.exit_code == 1
        assert "Use either --model <name> or --models" in result.output

    def test_model_name_backfill_on_pre_model_schema(self, tmp_path: Path) -> None:
        db = LineageDB(tmp_path / "lineage.db")
        db.init_schema()
        with db._engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE evaluation_runs DROP COLUMN model_name")
        db.init_schema()
        run_id = db.record_run(
            app_name="extraction",
            dataset_version="extraction_v1",
            configuration={"temperature": 0.1},
            aggregate_score=0.5,
            metric_results={"exact_match": 0.5},
            case_results=[],
            weights={},
        )
        run = db.get_run(run_id)
        assert run is not None
        assert run.model_name is None
        db.close()

    def test_compare_model_filter_shows_only_that_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("REFINELY_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
        monkeypatch.setattr("refinely.cli.context._client", lambda settings: StubLLMClient())
        monkeypatch.setattr("refinely.cli.context.get_registration", lambda app: _registration())
        db = LineageDB(tmp_path / "lineage.db")
        db.init_schema()
        db.record_run(
            app_name="extraction",
            dataset_version="extraction_v1",
            configuration={"temperature": 0.1},
            aggregate_score=0.5,
            metric_results={"exact_match": 0.5},
            case_results=[],
            weights={},
            model_name="gpt-4o",
        )
        db.record_run(
            app_name="extraction",
            dataset_version="extraction_v1",
            configuration={"temperature": 0.2},
            aggregate_score=0.9,
            metric_results={"exact_match": 0.9},
            case_results=[],
            weights={},
            model_name="claude-3",
        )
        db.close()

        result = _invoke(["compare", "extraction", "--model", "gpt-4o"], monkeypatch, tmp_path)
        assert result.exit_code == 0, result.output
        assert "gpt-4o" in result.output
        assert "0.5000" in result.output
        assert "0.9000" not in result.output

    def test_compare_model_no_matches_prints_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("REFINELY_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
        monkeypatch.setattr("refinely.cli.context._client", lambda settings: StubLLMClient())
        monkeypatch.setattr("refinely.cli.context.get_registration", lambda app: _registration())
        db = LineageDB(tmp_path / "lineage.db")
        db.init_schema()
        db.record_run(
            app_name="extraction",
            dataset_version="extraction_v1",
            configuration={"temperature": 0.1},
            aggregate_score=0.5,
            metric_results={"exact_match": 0.5},
            case_results=[],
            weights={},
            model_name="gpt-4o",
        )
        db.close()

        result = _invoke(["compare", "extraction", "--model", "nope-model"], monkeypatch, tmp_path)
        assert result.exit_code == 0, result.output
        assert "no runs found for that model" in result.output


class TestOptimizeAutosave:
    def test_optimize_saves_best_config(
        self, _config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from refinely.optimize.gate import GateResult, GateStats

        monkeypatch.setenv("REFINELY_LINEAGE_DB_PATH", str(tmp_path / "lineage.db"))
        monkeypatch.setattr("refinely.cli.context._client", lambda settings: StubLLMClient())
        monkeypatch.setattr("refinely.cli.context.get_registration", lambda app: _registration())
        monkeypatch.setattr("refinely.cli.context.run_study", lambda *a, **k: _FakeStudy())
        monkeypatch.setattr(
            "refinely.cli.optimize.gate_verdict",
            lambda b, c: GateResult(
                significant=True,
                baseline=GateStats(mean=0.0, std=0.0, n=3, ci_low=0.0, ci_high=0.0),
                candidate=GateStats(mean=0.7, std=0.0, n=3, ci_low=0.7, ci_high=0.7),
            ),
        )
        result = _invoke(["optimize", "extraction", "--trials", "2"], monkeypatch, tmp_path)
        assert result.exit_code == 0, result.output
        assert "opt-best.json" in result.output
        assert show_config("extraction", "opt-best") == {"temperature": 0.7}


class _FakeStudy:
    class _Trial:
        number: ClassVar[int] = 3
        value: ClassVar[float] = 0.8
        params: ClassVar[dict] = {"temperature": 0.7}

    trials: ClassVar[list] = [_Trial()]
    best_trial: ClassVar[Any] = _Trial()
