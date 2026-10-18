"""Application settings, loaded from environment variables / .env."""

import ipaddress
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The placeholder shipped in .env.example and docker-compose.yml. Anyone who has
# read either file can sign tokens with it, so production refuses to start on it.
INSECURE_JWT_SECRET = "dev-insecure-secret-change-me-in-prod-please"
MIN_JWT_SECRET_LENGTH = 32
MIN_PASSWORD_LENGTH = 12


# Suffixes reserved for private networks: never resolvable on the public
# internet (`.internal` is reserved by ICANN; host.docker.internal is the Docker
# host itself; `.local` is mDNS; `.localhost` is loopback).
_PRIVATE_SUFFIXES = (".internal", ".local", ".localhost")


def is_local_host(url: str) -> bool:
    """True for localhost, loopback/private IPs, single-label names (a compose
    service such as `ollama`) and reserved private suffixes: places where
    patient data stays on infrastructure you run."""
    host = urlsplit(url).hostname or ""
    if host == "localhost" or ("." not in host and host) or host.endswith(_PRIVATE_SUFFIXES):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback


class Settings(BaseSettings):
    # env_ignore_empty: docker-compose passes unset variables as "" (e.g.
    # RETENTION_DAYS: ${RETENTION_DAYS:-}); treat that as "not set", not as a value.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    # "production" turns misconfiguration into a startup error (weak JWT secret,
    # short bootstrap password), hides the interactive API docs and adds HSTS.
    app_env: Literal["development", "production"] = "development"

    llm_provider: str = "groq"
    groq_api_key: str | None = None
    # Must support tool calling (the evidence agent is a ReAct agent). A default that
    # the Groq org cannot access fails every fresh checkout and CI run with a 404, so
    # this is the model actually verified end to end, not a guess.
    groq_model: str = "openai/gpt-oss-20b"

    # LLM_PROVIDER=openai_compatible: a model you host (Ollama, vLLM, LM Studio)
    # behind an OpenAI-style /chat/completions API, e.g. http://localhost:11434/v1.
    # The model must support tool calling (the evidence agent is a ReAct agent).
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    # Prompts contain patient data. Production refuses a third-party model
    # (Groq, or an openai_compatible endpoint on a public host) unless this
    # asserts you have a data-processing agreement covering patient data.
    allow_external_llm_with_phi: bool = False

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

    # Distributed tracing (OpenTelemetry). Spans are always created -- they carry
    # the trace_id that correlates log lines -- but only exported when an OTLP/HTTP
    # endpoint is set (e.g. http://tempo:4318). Spans never carry patient data.
    otel_exporter_otlp_endpoint: str | None = None
    otel_console_exporter: bool = False
    otel_sample_ratio: float = 1.0

    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "medagent"

    jwt_secret_key: str = INSECURE_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Seeds exactly one admin account at startup if the users table is empty.
    # There is no public registration endpoint -- every other account must be
    # created by an admin via POST /admin/users.
    admin_bootstrap_username: str | None = None
    admin_bootstrap_password: str | None = None

    # Abuse limits, per process (like the circuit breakers). Login attempts are
    # limited per username and per client IP; agent runs (assess, follow-up,
    # drug check, WebSocket drafts) per user, because each one spends LLM quota.
    login_attempts_per_minute: float = 5.0
    login_attempts_burst: int = 5
    login_attempts_per_ip_per_minute: float = 20.0
    login_attempts_per_ip_burst: int = 20
    agent_runs_per_minute: float = 6.0
    agent_runs_burst: int = 3
    # Requests with a larger body are refused with 413 before they are parsed.
    max_request_body_bytes: int = 1_000_000

    # Retention for `medagent purge-expired`. Unset = nothing is ever deleted.
    # How long medical records and access logs must be kept is a legal decision
    # (often 7-10+ years); values under a year are refused as a likely mistake.
    retention_days: int | None = Field(default=None, ge=365)
    audit_retention_days: int | None = Field(default=None, ge=365)

    @model_validator(mode="after")
    def _refuse_insecure_production_config(self) -> Self:
        if self.app_env != "production":
            return self
        problems = []
        if (
            self.jwt_secret_key == INSECURE_JWT_SECRET
            or len(self.jwt_secret_key) < MIN_JWT_SECRET_LENGTH
        ):
            problems.append(
                f"JWT_SECRET_KEY must be a unique secret of at least {MIN_JWT_SECRET_LENGTH} "
                "characters (e.g. `openssl rand -hex 32`)"
            )
        if (
            self.admin_bootstrap_password is not None
            and len(self.admin_bootstrap_password) < MIN_PASSWORD_LENGTH
        ):
            problems.append(
                f"ADMIN_BOOTSTRAP_PASSWORD must be at least {MIN_PASSWORD_LENGTH} characters"
            )
        external_llm = self.llm_provider == "groq" or (
            self.llm_provider == "openai_compatible"
            and self.llm_base_url is not None
            and not is_local_host(self.llm_base_url)
        )
        if external_llm and not self.allow_external_llm_with_phi:
            problems.append(
                f"LLM_PROVIDER={self.llm_provider} sends patient data to a third party; use a "
                "self-hosted model (LLM_PROVIDER=openai_compatible with a local LLM_BASE_URL) "
                "or set ALLOW_EXTERNAL_LLM_WITH_PHI=true if you have a data agreement"
            )
        if problems:
            raise ValueError("Refusing to start with APP_ENV=production: " + "; ".join(problems))
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
