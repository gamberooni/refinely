from pathlib import Path

import pytest
from click.testing import CliRunner

from crucible.cli import main
from crucible.core.settings import Settings
from crucible.llm.usage import Result, TokenUsage
from crucible.registry import AppRegistration
from tests.stub_llm import StubLLMClient

DATASET_PATH = Path("datasets/extraction_v1.json")


@pytest.fixture
def _hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def _registration(*, dspy_factory=None, build_adapter=None) -> AppRegistration:
    return AppRegistration(
        name="extraction",
        build_adapter=build_adapter
        or (lambda client, settings, program_path=None: object()),
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
