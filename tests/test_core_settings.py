import pytest

from refinely.core.settings import Settings


@pytest.fixture(autouse=True)
def _no_real_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic against the repo-root .env (if present)."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REFINELY_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("REFINELY_MODEL_NAME", raising=False)
    monkeypatch.delenv("REFINELY_LINEAGE_DB_PATH", raising=False)
    monkeypatch.delenv("REFINELY_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setenv("REFINELY_OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("REFINELY_MODEL_NAME", "gpt-4o")
    monkeypatch.setenv("REFINELY_LINEAGE_DB_PATH", "/tmp/x/lineage.db")
    monkeypatch.setenv("REFINELY_BASE_URL", "https://gateway.example.com/v1")

    settings = Settings()

    assert settings.openai_api_key == "sk-test-123"
    assert settings.model_name == "gpt-4o"
    assert str(settings.lineage_db_path) == "/tmp/x/lineage.db"
    assert settings.base_url == "https://gateway.example.com/v1"
    assert settings.has_api_key


def test_settings_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REFINELY_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("REFINELY_MODEL_NAME", raising=False)
    monkeypatch.delenv("REFINELY_LINEAGE_DB_PATH", raising=False)
    monkeypatch.delenv("REFINELY_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings()

    assert settings.openai_api_key == ""
    assert settings.model_name == "deepseek-v4-flash"
    assert settings.base_url is None
    assert not settings.has_api_key


def test_settings_falls_back_to_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REFINELY_OPENAI_API_KEY", raising=False)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback-456")

    settings = Settings()

    assert settings.openai_api_key == "sk-fallback-456"
    assert settings.has_api_key


def test_settings_refinely_prefix_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REFINELY_OPENAI_API_KEY", "sk-prefixed")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback")

    settings = Settings()

    assert settings.openai_api_key == "sk-prefixed"


def test_settings_loads_from_env_file(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("REFINELY_OPENAI_API_KEY=sk-file-789\nREFINELY_MODEL_NAME=gpt-4o\n")

    settings = Settings(_env_file=str(env_file))

    assert settings.openai_api_key == "sk-file-789"
    assert settings.model_name == "gpt-4o"
    assert settings.has_api_key
