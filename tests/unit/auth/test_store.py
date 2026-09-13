from pathlib import Path

import pytest

from medagent.auth.security import hash_password
from medagent.auth.store import UserStore
from medagent.core.exceptions import AuthError
from medagent.memory.persistent import PersistentStore


async def _make_store(tmp_path: Path) -> UserStore:
    persistent = PersistentStore(str(tmp_path / "test.db"))
    await persistent.init_schema()
    return UserStore(persistent)


async def test_create_and_get_user_round_trips(tmp_path: Path) -> None:
    store = await _make_store(tmp_path)
    user = await store.create_user("dr.alpha", hash_password("s3cret!"))

    assert user.username == "dr.alpha"
    assert user.role == "doctor"

    record = await store.get_by_username("dr.alpha")
    assert record is not None
    fetched_user, hashed = record
    assert fetched_user.id == user.id
    assert hashed != "s3cret!"


async def test_get_missing_user_returns_none(tmp_path: Path) -> None:
    store = await _make_store(tmp_path)
    assert await store.get_by_username("nobody") is None


async def test_duplicate_username_raises_auth_error(tmp_path: Path) -> None:
    store = await _make_store(tmp_path)
    await store.create_user("dr.alpha", hash_password("s3cret!"))

    with pytest.raises(AuthError):
        await store.create_user("dr.alpha", hash_password("other"))
