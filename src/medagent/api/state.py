"""AppState: every long-lived client and agent, built once per app."""

from dataclasses import dataclass
from datetime import timedelta

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool
from starlette.requests import HTTPConnection

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.agents.trial_finder_agent import TrialFinderAgent
from medagent.auth.security import hash_password
from medagent.auth.store import UserStore
from medagent.core.config import Settings
from medagent.core.exceptions import AuthError, RateLimitedError
from medagent.core.interfaces import BaseKeyedLimiter, BaseLLMProvider
from medagent.infra.logging import get_logger
from medagent.infra.metrics import abuse_limit_hits_total
from medagent.infra.rate_limiter import KeyedRateLimiter
from medagent.llm.registry import ProviderRegistry
from medagent.memory import drafts
from medagent.memory.audit_log import AuditLogStore
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore
from medagent.memory.rate_limits import PostgresKeyedLimiter
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.reranker import CrossEncoderReranker
from medagent.rag.retriever import GuidelineRetrieverTool
from medagent.rag.vector_store import VectorStore
from medagent.tools.custom.interaction_checker import InteractionCheckerTool
from medagent.tools.mcp.healthcare import HealthcareMCPClient
from medagent.tools.mcp.http_base import HttpMCPClient
from medagent.tools.mcp.medical import MedicalMCPClient
from medagent.tools.mcp.research import ResearchMCPClient
from medagent.workflows.checkpointing import (
    build_checkpointer,
    open_postgres_checkpointer,
    purge_expired_drafts,
)
from medagent.workflows.followup import build_followup_graph

logger = get_logger(__name__)


@dataclass(frozen=True)
class Specialists:
    triage: TriageAgent
    drug_safety: DrugSafetyAgent
    evidence: EvidenceAgent
    trial_finder: TrialFinderAgent
    report: ReportAgent


def build_specialists(
    settings: Settings,
    llm: BaseLLMProvider,
    vector_store: VectorStore,
    embeddings: EmbeddingModel,
    *,
    medical: MedicalMCPClient,
    healthcare: HealthcareMCPClient,
    research: ResearchMCPClient,
) -> Specialists:
    """The agents and how they are wired: the app builds them here, and so
    does the agent eval (with its recording LLM and spy MCP clients), so the
    eval measures exactly what the app runs."""
    return Specialists(
        triage=TriageAgent(),
        drug_safety=DrugSafetyAgent(llm, InteractionCheckerTool(research)),
        evidence=EvidenceAgent(
            llm,
            medical,
            GuidelineRetrieverTool(
                embeddings,
                vector_store,
                reranker=(
                    CrossEncoderReranker(settings.rag_reranker_model)
                    if settings.rag_reranker_model
                    else None
                ),
                candidates=settings.rag_rerank_candidates,
                hybrid=settings.rag_hybrid,
            ),
        ),
        trial_finder=TrialFinderAgent(llm, healthcare),
        report=ReportAgent(),
    )


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.persistent_store = PersistentStore(settings.postgres_dsn)
        self.patient_store = PatientStore(self.persistent_store)
        self.user_store = UserStore(self.persistent_store)
        self.audit_log = AuditLogStore(self.persistent_store)

        # Abuse limits: per process, or shared by every replica (STATE_BACKEND).
        self.login_by_username = self._limiter(
            "login_user", settings.login_attempts_per_minute / 60, settings.login_attempts_burst
        )
        self.login_by_ip = self._limiter(
            "login_ip",
            settings.login_attempts_per_ip_per_minute / 60,
            settings.login_attempts_per_ip_burst,
        )
        self.agent_runs = self._limiter(
            "agent_runs", settings.agent_runs_per_minute / 60, settings.agent_runs_burst
        )

        self.llm = ProviderRegistry.get_provider(settings.llm_provider, settings)
        embeddings = EmbeddingModel()
        # Guideline vectors live in Postgres (pgvector) and share its pool.
        self.vector_store = VectorStore(self.persistent_store)
        medical_client = MedicalMCPClient(settings)
        healthcare_client = HealthcareMCPClient(settings)
        research_client = ResearchMCPClient(settings)
        # Held so /ready can probe them and report their circuit state.
        self.mcp_clients: dict[str, HttpMCPClient] = {
            "medical_mcp": medical_client,
            "healthcare_mcp": healthcare_client,
            "research_mcp": research_client,
        }

        specialists = build_specialists(
            settings,
            self.llm,
            self.vector_store,
            embeddings,
            medical=medical_client,
            healthcare=healthcare_client,
            research=research_client,
        )
        self.triage = specialists.triage
        self.drug_safety = specialists.drug_safety
        self.evidence = specialists.evidence
        self.trial_finder = specialists.trial_finder
        self.report = specialists.report

        # One long-lived graph with a checkpointer: the WebSocket approval flow
        # pauses in it between the doctor's message and their decision. The
        # Postgres checkpointer needs the event loop, so it is built in init().
        self._checkpoint_pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = None
        self.checkpointer: BaseCheckpointSaver
        self.followup_graph: CompiledStateGraph
        if settings.state_backend == "memory":
            self.checkpointer = build_checkpointer()
            self.followup_graph = self._build_followup_graph()

    def _limiter(self, name: str, rate_per_second: float, capacity: int) -> BaseKeyedLimiter:
        if self.settings.state_backend == "postgres":
            return PostgresKeyedLimiter(self.persistent_store, name, rate_per_second, capacity)
        return KeyedRateLimiter(rate_per_second, capacity)

    def _build_followup_graph(self) -> CompiledStateGraph:
        return build_followup_graph(
            self.triage,
            self.drug_safety,
            self.evidence,
            self.trial_finder,
            self.report,
            self.patient_store,
            checkpointer=self.checkpointer,
        )

    async def init(self) -> None:
        # Schema is owned by Alembic ("alembic upgrade head"), run before
        # startup in any real deployment; this just opens the connection pool.
        await self.persistent_store.init_schema()
        if self.settings.state_backend == "postgres":
            self.checkpointer, self._checkpoint_pool = await open_postgres_checkpointer(
                self.settings.postgres_dsn
            )
            self.followup_graph = self._build_followup_graph()
        await self._bootstrap_admin()

    async def purge_expired_drafts(self) -> None:
        ttl = timedelta(minutes=self.settings.approval_draft_ttl_minutes)
        if self.settings.state_backend == "postgres":
            await drafts.purge_expired_drafts(self.persistent_store, ttl)
        else:
            await purge_expired_drafts(self.checkpointer, ttl)

    async def _bootstrap_admin(self) -> None:
        """Seed exactly one admin account when the users table is empty.

        There is no public registration endpoint -- this is the only way an
        admin account can ever come into existence, so every other account
        must be created by an admin via POST /admin/users.
        """
        username = self.settings.admin_bootstrap_username
        password = self.settings.admin_bootstrap_password
        if not username or not password:
            return
        if await self.user_store.count_users() > 0:
            return

        try:
            await self.user_store.create_user(username, hash_password(password), role="admin")
        except AuthError:
            # Another replica starting at the same moment created it first.
            logger.info("admin_bootstrap_skipped", reason="already_created")
            return
        logger.info("admin_bootstrapped", username=username)

    async def close(self) -> None:
        if self._checkpoint_pool is not None:
            await self._checkpoint_pool.close()
        await self.persistent_store.close()


async def enforce_limit(limiter: BaseKeyedLimiter, key: str, *, limit: str, message: str) -> None:
    """Raises RateLimitedError (429 + Retry-After) when key is over its limit.
    `limit` names the limit for the metric; the key itself is never recorded."""
    wait = await limiter.try_acquire(key)
    if wait is not None:
        abuse_limit_hits_total.labels(limit=limit).inc()
        raise RateLimitedError(message, retry_after=wait)


def get_state(conn: HTTPConnection) -> AppState:
    """Works for both HTTP requests and WebSockets (HTTPConnection is their base)."""
    state: AppState = conn.app.state.medagent
    return state
