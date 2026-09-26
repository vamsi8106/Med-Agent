"""Application settings, loaded from environment variables / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "groq"
    groq_api_key: str | None = None
    # Must support tool calling (the evidence agent is a ReAct agent). A default that
    # the Groq org cannot access fails every fresh checkout and CI run with a 404, so
    # this is the model actually verified end to end, not a guess.
    groq_model: str = "openai/gpt-oss-20b"

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
    # Retry policy shared by the LLM and MCP clients. A rate limit whose
    # Retry-After exceeds retry_max_wait_seconds is not waited out (Groq's daily
    # cap says "try again in 8m" -- holding a doctor's request for that helps
    # nobody); retries stop once retry_budget_seconds have been spent in total.
    retry_max_wait_seconds: float = 10.0
    retry_budget_seconds: float = 45.0
    # Ceiling on cumulative LLM tokens (prompt + completion) a single
    # multi-agent run (one assess/followup call) may spend before remaining
    # specialist steps are skipped rather than run unboundedly.
    agent_max_tokens_per_run: int = 8000
    # Per-field cap on any single piece of external text (an MCP tool's raw
    # response, a literature/trial dump) folded into a prompt -- independent
    # of agent_max_tokens_per_run, which caps total spend across a whole run.
    agent_context_field_max_tokens: int = 1000
    # Final safety net on the fully-assembled synthesis prompt sent to the
    # LLM, in case several individually-capped fields still add up to too
    # much (e.g. many guideline hits concatenated into one citations block).
    agent_prompt_max_tokens: int = 4000
    # Each stored visit's assessment is the full markdown report it produced;
    # replaying five of those verbatim into every prompt is unbounded and
    # self-amplifying (each new report embeds the previous ones' evidence).
    agent_visit_summary_max_tokens: int = 250
    # A report drafted over the WebSocket waits in memory for the doctor's
    # decision, and survives a dropped connection so they can reconnect and
    # finish. Drafts hold patient data, so abandoned ones are purged after this.
    approval_draft_ttl_minutes: int = 30
    # Whole-loop deadline for a ReAct agent. LLM and MCP timeouts are per call
    # (30s each, with retries), so without this a hung provider can hold a
    # multi-turn loop for minutes. On expiry the agent falls back to its fixed
    # pipeline, which needs its own ~10-20s -- keep this well under the client's
    # patience. A ReAct evidence run measured ~15-25s.
    agent_loop_timeout_seconds: float = 40.0

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
