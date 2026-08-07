import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_prefix="REFINELY_", env_file=".env", extra="ignore")

    openai_api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    base_url: str | None = None
    model_name: str = "deepseek-v4-flash"
    lineage_db_path: Path = Path("lineage.db")

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)
