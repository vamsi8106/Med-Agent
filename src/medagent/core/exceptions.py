"""Exception hierarchy for MedAgent. Always raise a subclass, never bare Exception."""


class MedAgentError(Exception):
    """Base exception for all MedAgent errors."""


class ProviderError(MedAgentError):
    """Raised when an LLM provider call fails."""


class ToolError(MedAgentError):
    """Raised when a tool invocation fails."""


class MemoryError(MedAgentError):
    """Raised when patient memory/persistence operations fail."""


class MCPError(MedAgentError):
    """Raised when an MCP server call fails."""


class AuthError(MedAgentError):
    """Raised when authentication or authorization fails."""
