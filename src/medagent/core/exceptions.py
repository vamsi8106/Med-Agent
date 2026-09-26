"""Exception hierarchy for MedAgent. Always raise a subclass, never bare Exception.

Each subclass carries a `status_code` used by app.py's global exception
handler to map it to the right HTTP response -- callers never need to
translate exception type to status code by hand.
"""


class MedAgentError(Exception):
    """Base exception for all MedAgent errors."""

    status_code: int = 400


class ProviderError(MedAgentError):
    """Raised when an LLM provider call fails -- an upstream dependency, not
    a bad request, so it maps to 502 rather than 400."""

    status_code = 502


class ToolError(MedAgentError):
    """Raised when a tool invocation fails, typically due to invalid usage
    (e.g. missing required arguments) -- maps to 400."""

    status_code = 400


class MemoryError(MedAgentError):
    """Raised when patient memory/persistence operations fail -- a
    server-side failure, so it maps to 500 rather than 400."""

    status_code = 500


class MCPError(MedAgentError):
    """Raised when an MCP server call fails -- an upstream dependency, not
    a bad request, so it maps to 502 rather than 400."""

    status_code = 502


class AuthError(MedAgentError):
    """Raised when authentication or authorization fails -- maps to 401."""

    status_code = 401


class PatientNotFoundError(MedAgentError):
    """Raised when a workflow looks up a patient_id that doesn't exist --
    maps to 404 rather than the base class's 400."""

    status_code = 404


class AllAgentsFailedError(MedAgentError):
    """Raised when every specialist agent routed for a request failed, so there
    is nothing clinical to report -- an upstream dependency problem, so 502.
    A partial failure does not raise; it yields a degraded report instead."""

    status_code = 502


class AgentBudgetExceededError(MedAgentError):
    """Raised when a multi-agent run's cumulative LLM token usage would
    exceed its configured ceiling -- a resource-exhaustion condition, so it
    maps to 429 rather than 400."""

    status_code = 429
