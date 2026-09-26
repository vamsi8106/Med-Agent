import pytest

from medagent.auth.security import hash_password
from medagent.auth.store import UserStore
from medagent.core.exceptions import AuthError
from medagent.memory.persistent import PersistentStore


def _make_store(pg_dsn: str) -> UserStore:
    return UserStore(PersistentStore(pg_dsn))


async def test_create_and_get_user_round_trips(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    user = await store.create_user("dr.alpha", hash_password("s3cret!"))

    assert user.username == "dr.alpha"
    assert user.role == "doctor"

    record = await store.get_by_username("dr.alpha")
    assert record is not None
    fetched_user, hashed = record
    assert fetched_user.id == user.id
    assert hashed != "s3cret!"


async def test_get_missing_user_returns_none(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    assert await store.get_by_username("nobody") is None


async def test_duplicate_username_raises_auth_error(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    await store.create_user("dr.alpha", hash_password("s3cret!"))

    with pytest.raises(AuthError):
        await store.create_user("dr.alpha", hash_password("other"))


async def test_count_users(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    assert await store.count_users() == 0

    await store.create_user("dr.alpha", hash_password("s3cret!"))
    await store.create_user("dr.beta", hash_password("s3cret!"))

    assert await store.count_users() == 2


async def test_create_user_with_admin_role(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    user = await store.create_user("admin", hash_password("s3cret!"), role="admin")
    assert user.role == "admin"
