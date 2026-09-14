"""Shared fixtures: an ephemeral Postgres container for DB-backed unit tests.

Applies the exact same DDL as the Alembic migration (medagent.memory.schema)
directly via asyncpg, so tests never drift from what production runs via
`alembic upgrade head`. Requires a local Docker daemon; tests using
`pg_dsn`/`postgres_dsn` are skipped automatically if Docker isn't available,
mirroring how tests/integration/ skips when `npx` isn't on PATH.
"""

import asyncio
import os
import subprocess
import uuid
from collections.abc import Iterator

import asyncpg
import pytest

from medagent.memory.schema import (
    ADD_DOCTOR_ID_STATEMENTS,
    AUDIT_LOG_CREATE_STATEMENTS,
    CREATE_STATEMENTS,
    TABLES_IN_DEPENDENCY_ORDER,
)

_CANDIDATE_DOCKER_HOSTS = [os.environ.get("DOCKER_HOST"), None, "unix:///var/run/docker.sock"]


def _find_working_docker_host() -> str | None:
    """Returns the DOCKER_HOST to use, or None if no candidate works.

    Note: a successful "use the default context" candidate (DOCKER_HOST
    unset) is itself represented as None in _CANDIDATE_DOCKER_HOSTS, which
    would be indistinguishable from "not found" if returned as-is -- return
    "" instead so callers can still tell success from failure via `is None`.
    """
    for host in _CANDIDATE_DOCKER_HOSTS:
        env = os.environ.copy()
        if host:
            env["DOCKER_HOST"] = host
        try:
            result = subprocess.run(["docker", "version"], env=env, capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return host or ""
    return None


async def _wait_until_ready_and_apply_schema(dsn: str) -> None:
    last_error: Exception | None = None
    for _ in range(30):
        try:
            conn = await asyncpg.connect(dsn)
            break
        except Exception as exc:  # noqa: BLE001 - retrying until Postgres accepts connections
            last_error = exc
            await asyncio.sleep(1)
    else:
        raise RuntimeError(f"Postgres test container never became ready: {last_error}")

    try:
        for statement in [
            *CREATE_STATEMENTS,
            *AUDIT_LOG_CREATE_STATEMENTS,
            *ADD_DOCTOR_ID_STATEMENTS,
        ]:
            await conn.execute(statement)
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def postgres_dsn() -> Iterator[str]:
    working_host = _find_working_docker_host()
    if working_host is None:
        pytest.skip("Docker is not available; skipping Postgres-backed tests")

    env = os.environ.copy()
    if working_host:
        env["DOCKER_HOST"] = working_host

    container_name = f"medagent-test-pg-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "-e",
            "POSTGRES_USER=medagent",
            "-e",
            "POSTGRES_PASSWORD=medagent",
            "-e",
            "POSTGRES_DB=medagent_test",
            "-p",
            "0:5432",
            "postgres:16-alpine",
        ],
        env=env,
        check=True,
        capture_output=True,
    )

    try:
        port_output = subprocess.run(
            ["docker", "port", container_name, "5432/tcp"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        host_port = port_output.splitlines()[0].rsplit(":", 1)[-1]
        dsn = f"postgresql://medagent:medagent@localhost:{host_port}/medagent_test"
        asyncio.run(_wait_until_ready_and_apply_schema(dsn))
        yield dsn
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], env=env, capture_output=True)


@pytest.fixture
async def pg_dsn(postgres_dsn: str) -> str:
    conn = await asyncpg.connect(postgres_dsn)
    try:
        for table in TABLES_IN_DEPENDENCY_ORDER:
            await conn.execute(f"TRUNCATE TABLE {table} CASCADE")
    finally:
        await conn.close()
    return postgres_dsn
