from unittest.mock import AsyncMock, patch

from medagent.api.state import AppState
from medagent.auth.store import UserStore
from medagent.core.config import Settings
from medagent.memory.persistent import PersistentStore


async def test_a_replica_that_loses_the_bootstrap_race_starts_anyway(pg_dsn: str) -> None:
    """Two replicas starting on an empty database both see zero users; the
    second insert hits the unique username and must not crash startup."""
    store = PersistentStore(pg_dsn)
    settings = Settings(
        _env_file=None,
        postgres_dsn=pg_dsn,
        admin_bootstrap_username="admin",
        admin_bootstrap_password="admin-pass-1234",
    )
    state = AppState.__new__(AppState)
    state.settings = settings
    state.user_store = UserStore(store)

    with patch.object(UserStore, "count_users", AsyncMock(return_value=0)):
        await state._bootstrap_admin()
        await state._bootstrap_admin()

    assert [user.username for user in await state.user_store.list_users()] == ["admin"]
    await store.close()


def test_build_specialists_wires_the_agents_to_the_clients_it_is_given() -> None:
    from unittest.mock import MagicMock

    from medagent.api.state import build_specialists
    from medagent.llm.mock_provider import MockLLMProvider
    from medagent.tools.mcp.healthcare import HealthcareMCPClient
    from medagent.tools.mcp.medical import MedicalMCPClient
    from medagent.tools.mcp.research import ResearchMCPClient

    settings = Settings(_env_file=None, rag_hybrid=True, rag_rerank_candidates=7)
    llm = MockLLMProvider()
    medical, healthcare, research = (
        MedicalMCPClient(settings),
        HealthcareMCPClient(settings),
        ResearchMCPClient(settings),
    )
    vector_store, embeddings = MagicMock(), MagicMock()

    agents = build_specialists(
        settings,
        llm,
        vector_store,
        embeddings,
        medical=medical,
        healthcare=healthcare,
        research=research,
    )

    assert agents.drug_safety._llm is llm
    assert agents.drug_safety._interaction_checker._research_client is research
    assert agents.evidence._medical_client is medical
    assert agents.trial_finder._healthcare_client is healthcare
    retriever = agents.evidence._guideline_retriever
    assert retriever._vector_store is vector_store
    assert retriever._hybrid is True
    assert retriever._candidates == 7
    assert retriever._reranker is None  # RAG_RERANKER_MODEL unset
