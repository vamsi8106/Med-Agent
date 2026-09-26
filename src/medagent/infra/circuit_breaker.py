"""Circuit breaker: fails fast when a dependency (an MCP server, the LLM) is down.

Only failures that indicate an unhealthy upstream count toward opening it:
timeouts, connection failures and 5xx. A 4xx or a rate limit means the server is
up and answering, so a burst of bad requests must not take it offline.
"""

import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import ParamSpec, TypeVar

from opentelemetry import trace

from medagent.core.exceptions import MCPError, UpstreamError
from medagent.infra.metrics import circuit_breaker_opens_total, circuit_breaker_state

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


_STATE_VALUE = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}


def _default_counts_failure(exc: Exception) -> bool:
    return bool(getattr(exc, "trips_breaker", True))


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        *,
        name: str = "unknown",
        error_type: type[UpstreamError] = MCPError,
        count_failure: Callable[[Exception], bool] = _default_counts_failure,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._name = name
        self._error_type = error_type
        self._count_failure = count_failure
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None
        circuit_breaker_state.labels(target=name).set(0)

    @property
    def state(self) -> CircuitState:
        if self._state is CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._publish()
        return self._state

    async def call(self, func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
        if self.state is CircuitState.OPEN:
            trace.get_current_span().add_event(
                "circuit_open", {"medagent.breaker.target": self._name}
            )
            raise self._error_type(
                f"Circuit breaker for {self._name} is open; refusing the call to protect it",
                reason="circuit_open",
            )

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            if self._count_failure(exc):
                self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _publish(self) -> None:
        circuit_breaker_state.labels(target=self._name).set(_STATE_VALUE[self._state])

    def _record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            if self._state is not CircuitState.OPEN:
                circuit_breaker_opens_total.labels(target=self._name).inc()
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            self._publish()

    def _record_success(self) -> None:
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None
        self._publish()
