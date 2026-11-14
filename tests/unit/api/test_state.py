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
