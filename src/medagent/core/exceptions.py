"""Exception hierarchy for MedAgent. Always raise a subclass, never bare Exception.

Each subclass carries a `status_code` used by app.py's global exception
handler to map it to the right HTTP response -- callers never need to
translate exception type to status code by hand.
"""

import math


class MedAgentError(Exception):
    """Base exception for all MedAgent errors."""

    status_code: int = 400


def wait_phrase(retry_after: float | None) -> str:
    """ "shortly" / "in about 8 seconds" / "in about 8 minutes" -- how long an
    upstream asked us to wait, worded for a doctor."""
    if retry_after is None:
        return "shortly"
    seconds = max(1, math.ceil(retry_after))
    if seconds < 90:
        return f"in about {seconds} seconds"
    return f"in about {math.ceil(seconds / 60)} minutes"


# Reasons whose failures say the upstream itself is unhealthy. Rate limits and
# client errors (bad request, unknown model) do not: the server is up and
# answering, so they must never open a circuit breaker.
_BREAKER_TRIPPING_REASONS = frozenset({"timeout", "server_error", "connection_error"})


class UpstreamError(MedAgentError):
    """A failure calling a dependency we don't control (the LLM, an MCP server).

    Carries what the retry policy and circuit breaker need to act correctly:
    `reason` classifies it, `retryable` says whether trying again can help
    (default False -- an unclassified error is never blindly retried), and
    `retry_after` is the wait in seconds the upstream itself asked for. An
    upstream problem, not a bad request, so it maps to 502 -- except a rate
    limit, which is 429: the service is up and we are being throttled, and the
    client should back off for `retry_after`, not treat it as an outage.
    """

    status_code = 502
    service = "an upstream service"

    def __init__(
        self,
        message: str = "",
        *,
        reason: str = "error",
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.retryable = retryable
        self.retry_after = retry_after
        if reason == "rate_limited":
            self.status_code = 429

    @property
    def trips_breaker(self) -> bool:
        return self.reason in _BREAKER_TRIPPING_REASONS

    def public_message(self) -> str:
        """Fixed, client-safe text. The raw message can carry account ids (Groq
        embeds the organisation id) or internal hostnames (MCP URLs), so it is
        logged server-side and never returned."""
        if self.reason == "rate_limited":
            return f"{self.service} is rate-limited; try again {wait_phrase(self.retry_after)}"
        return f"{self.service} was unavailable"


class ProviderError(UpstreamError):
    """Raised when an LLM provider call fails."""

    service = "the language model service"


class ToolError(MedAgentError):
    """Raised when a tool invocation fails, typically due to invalid usage
    (e.g. missing required arguments) -- maps to 400."""

    status_code = 400


class MemoryError(MedAgentError):
    """Raised when patient memory/persistence operations fail -- a
    server-side failure, so it maps to 500 rather than 400."""

    status_code = 500


class MCPError(UpstreamError):
    """Raised when an MCP server call fails."""

    service = "an external medical data source"


class AuthError(MedAgentError):
    """Raised when authentication or authorization fails -- maps to 401."""

    status_code = 401


class PatientNotFoundError(MedAgentError):
    """Raised when a workflow looks up a patient_id that doesn't exist --
    maps to 404 rather than the base class's 400."""

    status_code = 404


class AllAgentsFailedError(MedAgentError):
    """Raised when every specialist agent routed for a request failed, so there
    is nothing clinical to report -- an upstream dependency problem, so 502, or
    429 with `retry_after` when they were all rate-limited. A partial failure
    does not raise; it yields a degraded report instead."""

    status_code = 502

    def __init__(
        self, message: str = "", *, reason: str = "error", retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.retry_after = retry_after
        if reason == "rate_limited":
            self.status_code = 429


class AgentBudgetExceededError(MedAgentError):
    """Raised when a multi-agent run's cumulative LLM token usage would
    exceed its configured ceiling -- a resource-exhaustion condition, so it
    maps to 429 rather than 400."""

    status_code = 429
