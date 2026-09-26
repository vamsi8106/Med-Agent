"""AppState: every long-lived client and agent, built once per app."""

from starlette.requests import HTTPConnection

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.agents.trial_finder_agent import TrialFinderAgent
from medagent.auth.security import hash_password
from medagent.auth.store import UserStore
from medagent.core.config import Settings
from medagent.infra.logging import get_logger
from medagent.llm.registry import ProviderRegistry
from medagent.memory.audit_log import AuditLogStore
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.retriever import GuidelineRetrieverTool
from medagent.rag.vector_store import VectorStore
from medagent.tools.custom.interaction_checker import InteractionCheckerTool
from medagent.tools.mcp.healthcare import HealthcareMCPClient
from medagent.tools.mcp.http_base import HttpMCPClient
from medagent.tools.mcp.medical import MedicalMCPClient
from medagent.tools.mcp.research import ResearchMCPClient
from medagent.workflows.checkpointing import build_checkpointer
from medagent.workflows.followup import build_followup_graph

logger = get_logger(__name__)


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.persistent_store = PersistentStore(settings.postgres_dsn)
        self.patient_store = PatientStore(self.persistent_store)
        self.user_store = UserStore(self.persistent_store)
        self.audit_log = AuditLogStore(self.persistent_store)

        self.llm = ProviderRegistry.get_provider(settings.llm_provider, settings)
        embeddings = EmbeddingModel()
        self.vector_store = VectorStore(settings.chroma_host, settings.chroma_port)
        medical_client = MedicalMCPClient(settings)
        healthcare_client = HealthcareMCPClient(settings)
        research_client = ResearchMCPClient(settings)
        # Held so /ready can probe them and report their circuit state.
        self.mcp_clients: dict[str, HttpMCPClient] = {
            "medical_mcp": medical_client,
            "healthcare_mcp": healthcare_client,
            "research_mcp": research_client,
        }

        self.triage = TriageAgent()
        self.drug_safety = DrugSafetyAgent(self.llm, InteractionCheckerTool(research_client))
        self.evidence = EvidenceAgent(
            self.llm, medical_client, GuidelineRetrieverTool(embeddings, self.vector_store)
        )
        self.trial_finder = TrialFinderAgent(self.llm, healthcare_client)
        self.report = ReportAgent()

        # One long-lived graph with a checkpointer: the WebSocket approval flow
        # pauses in it between the doctor's message and their decision.
        self.checkpointer = build_checkpointer()
        self.followup_graph = build_followup_graph(
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
        await self._bootstrap_admin()

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

        await self.user_store.create_user(username, hash_password(password), role="admin")
        logger.info("admin_bootstrapped", username=username)

    async def close(self) -> None:
        await self.persistent_store.close()


def get_state(conn: HTTPConnection) -> AppState:
    """Works for both HTTP requests and WebSockets (HTTPConnection is their base)."""
    state: AppState = conn.app.state.medagent
    return state
