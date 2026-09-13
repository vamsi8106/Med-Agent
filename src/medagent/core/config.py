"""Application settings, loaded from environment variables / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "groq"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    log_level: str = "INFO"

    medical_mcp_command: str = "npx"
    medical_mcp_args: list[str] = ["-y", "medical-mcp"]
    healthcare_mcp_command: str = "node"
    healthcare_mcp_args: list[str] = ["./vendor/healthcare-mcp/build/index.js"]
    research_mcp_command: str = "node"
    research_mcp_args: list[str] = ["./vendor/med-research-mcp-suite/dist/index.js"]

    postgres_dsn: str = "postgresql://medagent:medagent@localhost:5432/medagent"
    chroma_persist_dir: str = ".chroma"

    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "medagent"

    jwt_secret_key: str = "dev-insecure-secret-change-me-in-prod-please"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
