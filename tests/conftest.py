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
    ADD_RECORD_HISTORY_STATEMENTS,
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
            *ADD_RECORD_HISTORY_STATEMENTS,
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


# --- retry-policy test helpers ---------------------------------------------------


class FakeClock:
    """Replaces retry's sleep and clock: sleeping advances a fake clock instantly
    and records the delay, so a policy that would wait for minutes runs in
    microseconds and its exact behaviour can be asserted."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr("medagent.infra.retry._sleep", fake.sleep)
    monkeypatch.setattr("medagent.infra.retry._now", fake.monotonic)
    return fake


def metric_value(name: str, **labels: str) -> float:
    """Current value of a Prometheus sample (0 if it hasn't been touched yet).
    Metrics are process-global, so tests compare before/after, not absolutes."""
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name, labels) or 0.0


# --- telemetry test helpers ---------------------------------------------------------
#
# OpenTelemetry allows the global tracer provider to be set once per process, so
# one in-memory provider is installed here for the whole session. Tests read the
# spans it captured through the `spans` fixture. (configure_tracing() never
# overrides an installed provider, so the app's lifespan leaves this one alone.)

import sys  # noqa: E402
from typing import Any  # noqa: E402

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

_SPAN_EXPORTER = InMemorySpanExporter()
_SESSION_PROVIDER = TracerProvider()
_SESSION_PROVIDER.add_span_processor(SimpleSpanProcessor(_SPAN_EXPORTER))
trace.set_tracer_provider(_SESSION_PROVIDER)


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    _SPAN_EXPORTER.clear()
    yield _SPAN_EXPORTER
    _SPAN_EXPORTER.clear()


def span_text(span: Any) -> str:
    """Everything a span exposes -- name, attributes, events (and their
    attributes) -- flattened to one lowercase string, for no-PHI assertions."""
    parts = [span.name, *(f"{k}={v}" for k, v in (span.attributes or {}).items())]
    for event in span.events:
        parts.append(event.name)
        parts.extend(f"{k}={v}" for k, v in (event.attributes or {}).items())
    return " ".join(str(p) for p in parts).lower()


class _RecordingLogger:
    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self._sink = sink

    def _log(self, event: str, **fields: Any) -> None:
        self._sink.append({"event": event, **fields})

    info = warning = error = debug = exception = _log


@pytest.fixture
def recorded_logs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every log call any medagent module makes, as dicts. Replaces each
    module's `logger` directly, so it works however structlog has been
    configured or cached by earlier tests."""
    sink: list[dict[str, Any]] = []
    recorder = _RecordingLogger(sink)
    for name, module in list(sys.modules.items()):
        if name.startswith("medagent.") and hasattr(module, "logger"):
            monkeypatch.setattr(module, "logger", recorder)
    return sink
