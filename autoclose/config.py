"""Application configuration and environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider: "openai" | "gemini"
    llm_provider: Literal["openai", "gemini"] = "openai"

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Google Gemini Configuration
    google_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # Vector Store
    chroma_persist_directory: str = "./data/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Database (MCP simulation)
    database_path: str = "./data/autoclose.db"

    # File storage
    upload_directory: str = "./data/uploads"
    processed_directory: str = "./data/processed"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Notification (simulated webhook)
    notification_webhook_url: str = ""

    @property
    def llm_api_key(self) -> str:
        """Return the appropriate API key for the configured provider."""
        if self.llm_provider == "openai":
            return self.openai_api_key
        return self.google_api_key


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
