"""Tool contract and result wrapper for MedAgent's tool layer."""

from typing import Any

from pydantic import BaseModel

from medagent.core.interfaces import BaseTool

__all__ = ["BaseTool", "ToolResult"]


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: Any | None = None
    error: str | None = None
    # Class name of the MedAgentError behind a failure, so a caller can tell an
    # unreachable MCP server (MCPError) from bad input (ToolError) instead of
    # collapsing every failure into one generic message.
    error_type: str | None = None
