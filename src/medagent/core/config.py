"""Application settings, loaded from environment variables / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "groq"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    log_level: str = "INFO"

    database_path: str = "medagent.db"
    chroma_persist_dir: str = ".chroma"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
