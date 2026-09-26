"""Async retry with exponential backoff, jitter, and upstream-aware waits.

Only errors that say they are worth retrying are retried. A permanent failure
(bad request, unknown model) fails on the first attempt instead of burning
attempts, and when the upstream itself says how long to wait (Retry-After) that
is honoured -- unless it is longer than `max_wait`, in which case there is no
point holding a doctor's request open and the error is raised straight away.
`budget` caps the total time spent so a run of slow failures cannot stack up.
"""

import asyncio
import functools
import random
import time
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from opentelemetry import trace

from medagent.core.exceptions import UpstreamError
from medagent.infra.metrics import retry_attempts_total

P = ParamSpec("P")
T = TypeVar("T")

# Indirections so tests can run the policy instantly and deterministically.
_sleep = asyncio.sleep
_now = time.monotonic


def _delay(exc: Exception, attempt: int, base_delay: float, max_delay: float) -> float:
    asked = getattr(exc, "retry_after", None)
    if asked is not None:
        return float(asked) + random.uniform(0, min(asked * 0.1, 0.5))
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    return delay + random.uniform(0, delay * 0.1)


def retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: tuple[type[Exception], ...] = (UpstreamError,),
    max_wait: float | None = None,
    budget: float | None = None,
    target: str = "unknown",
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            started = _now()
            attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    attempt += 1
                    if attempt >= max_attempts or not getattr(exc, "retryable", True):
                        raise
                    asked = getattr(exc, "retry_after", None)
                    if max_wait is not None and asked is not None and asked > max_wait:
                        raise
                    delay = _delay(exc, attempt, base_delay, max_delay)
                    if budget is not None and _now() - started + delay > budget:
                        raise
                    reason = getattr(exc, "reason", "error")
                    retry_attempts_total.labels(target=target, reason=reason).inc()
                    # An event on the current span, so a trace shows *why* a call
                    # was slow (a rate limit waited out, a timeout retried).
                    trace.get_current_span().add_event(
                        "retry",
                        {
                            "medagent.retry.target": target,
                            "medagent.retry.reason": reason,
                            "medagent.retry.attempt": attempt,
                            "medagent.retry.delay_s": round(delay, 3),
                        },
                    )
                    await _sleep(delay)

        return wrapper

    return decorator
