"""Application settings, loaded from environment variables / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "groq"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    log_level: str = "INFO"

    # Each MCP server runs as its own container reachable over HTTP: medical-mcp
    # (stdio-only) sits behind a stdio<->HTTP bridge sidecar; healthcare-mcp and
    # med-research-mcp-suite expose their own REST APIs directly. Defaults match
    # the service DNS names used in docker-compose.yml / k8s Service names.
    medical_mcp_url: str = "http://medical-mcp-bridge:8080"
    healthcare_mcp_url: str = "http://healthcare-mcp:3000"
    research_mcp_url: str = "http://research-mcp:3000"
    mcp_timeout_seconds: float = 30.0
    # Free medical APIs behind these MCP servers have low rate ceilings --
    # this throttles outbound calls per server before they ever hit retry.
    mcp_rate_limit_per_second: float = 5.0
    mcp_rate_limit_capacity: int = 10

    llm_timeout_seconds: float = 30.0

    postgres_dsn: str = "postgresql://medagent:medagent@localhost:5432/medagent"
    # ChromaDB runs as its own server container so a persist-dir volume can be
    # shared safely across multiple app replicas instead of embedded per-pod.
    chroma_host: str = "chromadb"
    chroma_port: int = 8000

    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "medagent"

    jwt_secret_key: str = "dev-insecure-secret-change-me-in-prod-please"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Seeds exactly one admin account at startup if the users table is empty.
    # There is no public registration endpoint -- every other account must be
    # created by an admin via POST /admin/users.
    admin_bootstrap_username: str | None = None
    admin_bootstrap_password: str | None = None


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
