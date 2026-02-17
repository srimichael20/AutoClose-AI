"""Application configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-based settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["openai", "gemini"] = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    google_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    @field_validator("openai_api_key", "google_api_key", mode="before")
    @classmethod
    def strip_api_key(cls, v) -> str:
        """Strip whitespace and quotes that often break API keys."""
        if v is None:
            return ""
        return str(v).strip().strip('"\'')

    chroma_persist_directory: str = "./data/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"

    database_path: str = "./data/autoclose.db"
    upload_directory: str = "./data/uploads"
    processed_directory: str = "./data/processed"

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    notification_webhook_url: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
