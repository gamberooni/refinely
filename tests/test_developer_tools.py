"""Tests for dev-ergonomics: `new app`, `doctor`, and `dataset stats`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from refinely.cli import main
from refinely.core.settings import Settings
from refinely.devtools.doctor import CheckResult, run_checks
from refinely.devtools.scaffold import ScaffoldError, write_app
from refinely.eval.datasets import dataset_stats
from refinely.registry import AppRegistration


@pytest.fixture
def _hermetic_settings() -> None:
    pytest.importorskip("refinely")
    import refinely

    refinely.core.settings.Settings.model_config["env_file"] = None
    yield


class TestScaffold:
    def test_write_app_creates_module_and_stub(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("refinely.devtools.scaffold.APPS_DIR", tmp_path / "apps")
        monkeypatch.setattr("refinely.devtools.scaffold.DATASETS_DIR", tmp_path / "datasets")

        app_path, stub_path = write_app("summarize")

        assert app_path == tmp_path / "apps" / "summarize.py"
        assert stub_path == tmp_path / "datasets" / "summarize_v1.json"
        assert app_path.exists()
        assert stub_path.exists()
        stub = json.loads(stub_path.read_text())
        assert stub == {"version": "summarize_v1", "cases": []}
        module = app_path.read_text()
        assert "register_app" in module
        assert "summarize" in module
        assert "summarize_v1.json" in module

    def test_write_app_with_explicit_dataset_skips_stub(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("refinely.devtools.scaffold.APPS_DIR", tmp_path / "apps")
        monkeypatch.setattr("refinely.devtools.scaffold.DATASETS_DIR", tmp_path / "datasets")
        custom = tmp_path / "my_ds.json"
        custom.write_text(json.dumps({"version": "v9", "cases": []}))

        app_path, stub_path = write_app("summarize", dataset_path=custom)

        assert stub_path is None
        assert app_path.exists()
        assert (tmp_path / "datasets" / "summarize_v1.json").exists() is False
        assert "my_ds.json" in app_path.read_text()

    def test_write_app_rejects_invalid_name(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("refinely.devtools.scaffold.APPS_DIR", tmp_path / "apps")
        monkeypatch.setattr("refinely.devtools.scaffold.DATASETS_DIR", tmp_path / "datasets")

        with pytest.raises(ScaffoldError, match="valid Python identifier"):
            write_app("1bad")

    def test_write_app_rejects_reserved_name(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("refinely.devtools.scaffold.APPS_DIR", tmp_path / "apps")
        monkeypatch.setattr("refinely.devtools.scaffold.DATASETS_DIR", tmp_path / "datasets")

        with pytest.raises(ScaffoldError, match="reserved"):
            write_app("refinely")

    def test_write_app_refuses_existing_module(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("refinely.devtools.scaffold.APPS_DIR", tmp_path / "apps")
        monkeypatch.setattr("refinely.devtools.scaffold.DATASETS_DIR", tmp_path / "datasets")
        (tmp_path / "apps").mkdir(parents=True)
        (tmp_path / "apps" / "summarize.py").write_text("x")

        with pytest.raises(ScaffoldError, match="already exists"):
            write_app("summarize")

    def test_write_app_refuses_existing_stub(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("refinely.devtools.scaffold.APPS_DIR", tmp_path / "apps")
        monkeypatch.setattr("refinely.devtools.scaffold.DATASETS_DIR", tmp_path / "datasets")
        (tmp_path / "datasets").mkdir(parents=True)
        (tmp_path / "datasets" / "summarize_v1.json").write_text("{}")

        with pytest.raises(ScaffoldError, match="already exists"):
            write_app("summarize")


class TestDoctor:
    def test_run_checks_default_has_no_network(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "refinely.devtools.doctor._check_apps", lambda: CheckResult("apps", True, "ok")
        )
        monkeypatch.setattr(
            "refinely.devtools.doctor._check_datasets", lambda: CheckResult("datasets", True, "ok")
        )
        monkeypatch.setattr(
            "refinely.devtools.doctor._check_schema", lambda s: CheckResult("schema", True, "ok")
        )
        monkeypatch.setattr(
            "refinely.devtools.doctor._check_env", lambda s: CheckResult("env", True, "ok")
        )

        results = run_checks(Settings(), network=False)
        names = {r.name for r in results}
        assert "network" not in names

    def test_run_checks_network_only_when_flagged(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "refinely.devtools.doctor._check_apps", lambda: CheckResult("apps", True, "ok")
        )
        monkeypatch.setattr(
            "refinely.devtools.doctor._check_datasets", lambda: CheckResult("datasets", True, "ok")
        )
        monkeypatch.setattr(
            "refinely.devtools.doctor._check_schema", lambda s: CheckResult("schema", True, "ok")
        )
        monkeypatch.setattr(
            "refinely.devtools.doctor._check_env", lambda s: CheckResult("env", True, "ok")
        )
        monkeypatch.setattr(
            "refinely.devtools.doctor._check_network",
            lambda s: CheckResult("network", True, "reachable"),
        )

        results = run_checks(Settings(), network=True)
        assert any(r.name == "network" for r in results)


class TestDatasetStats:
    def _write_dataset(self, path: Path) -> None:
        data = {
            "version": "v1",
            "cases": [
                {"id": "c0", "input": {"text": "a", "n": 1}, "expected": {"label": "x"}},
                {"id": "c1", "input": {"text": "b", "n": 2}, "expected": {"label": "y"}},
                {"id": "c2", "input": {"text": "c", "n": 3}, "expected": {"label": "z"}},
            ],
        }
        path.write_text(json.dumps(data))

    def test_dataset_stats_counts_and_shapes(self, tmp_path: Path) -> None:
        path = tmp_path / "ds.json"
        self._write_dataset(path)

        stats = dataset_stats(path)

        assert stats.case_count == 3
        assert stats.file_size_bytes > 0
        assert stats.input_field_counts == {"text": 3, "n": 3}
        assert stats.expected_shape_counts == {"dict": 3}
        assert stats.expected_key_counts == {"label": 3}
        assert stats.malformed == []

    def test_dataset_stats_flags_malformed(self, tmp_path: Path) -> None:
        path = tmp_path / "ds.json"
        path.write_text(
            json.dumps(
                {
                    "version": "v1",
                    "cases": [
                        {"id": "c0", "input": {"text": "a"}, "expected": {"label": "x"}},
                        {"id": "c1", "input": {"text": "b"}, "expected": {"label": "y"}},
                        {"id": "c2", "input": {"other": "z"}, "expected": "not-a-dict"},
                    ],
                }
            )
        )

        stats = dataset_stats(path)

        assert stats.case_count == 3
        assert stats.malformed == ["c2"]

    def test_dataset_stats_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="not found"):
            dataset_stats(tmp_path / "nope.json")


class TestDeveloperCli:
    def _invoke(self, args: list[str]) -> object:
        return CliRunner().invoke(main, args, env={"COLUMNS": "200"})

    def test_new_app_command(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("refinely.devtools.scaffold.APPS_DIR", tmp_path / "apps")
        monkeypatch.setattr("refinely.devtools.scaffold.DATASETS_DIR", tmp_path / "datasets")

        result = self._invoke(["new", "app", "summarize"])

        assert result.exit_code == 0, result.output
        assert "Created" in result.output
        assert "refinely.apps" in result.output
        assert 'summarize = "apps.summarize"' in result.output
        assert (tmp_path / "apps" / "summarize.py").exists()
        assert (tmp_path / "datasets" / "summarize_v1.json").exists()

    def test_new_app_invalid_name(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("refinely.devtools.scaffold.APPS_DIR", tmp_path / "apps")
        monkeypatch.setattr("refinely.devtools.scaffold.DATASETS_DIR", tmp_path / "datasets")

        result = self._invoke(["new", "app", "1bad"])

        assert result.exit_code == 1
        assert "valid Python identifier" in result.output

    def test_doctor_all_pass_exits_zero(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "refinely.cli.devtools.run_checks",
            lambda settings, network=False: [
                CheckResult("apps", True, "ok"),
                CheckResult("datasets", True, "ok"),
                CheckResult("schema", True, "ok"),
                CheckResult("env", True, "ok"),
            ],
        )

        result = self._invoke(["doctor"])

        assert result.exit_code == 0
        assert "all checks passed" in result.output

    def test_doctor_failure_exits_one_with_hint(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "refinely.cli.devtools.run_checks",
            lambda settings, network=False: [
                CheckResult("env", False, "REFINELY_OPENAI_API_KEY is not set", hint="set it"),
            ],
        )

        result = self._invoke(["doctor"])

        assert result.exit_code == 1
        assert "env" in result.output
        assert "hint: set it" in result.output

    def test_dataset_stats_command(self, tmp_path: Path, monkeypatch) -> None:
        ds_path = tmp_path / "ds.json"
        ds_path.write_text(
            json.dumps(
                {
                    "version": "v1",
                    "cases": [
                        {"id": "c0", "input": {"text": "a"}, "expected": {"label": "x"}},
                        {"id": "c1", "input": {"text": "b"}, "expected": {"label": "y"}},
                    ],
                }
            )
        )
        monkeypatch.setattr(
            "refinely.cli.context.get_registration",
            lambda app: AppRegistration(
                name="extraction",
                build_adapter=lambda client, settings=None: object(),
                metrics_factory=lambda client, settings=None: [],
                search_space=lambda trial: {},
                default_config={},
                weights={},
                dataset_path=ds_path,
            ),
        )

        result = self._invoke(["dataset", "extraction"])

        assert result.exit_code == 0, result.output
        assert "cases:    2" in result.output
        assert "text=2" in result.output
        assert "dict=2" in result.output
