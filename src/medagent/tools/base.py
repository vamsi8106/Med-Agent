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
