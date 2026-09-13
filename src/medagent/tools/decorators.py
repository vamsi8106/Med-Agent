"""@tool decorator: wraps a function into a BaseTool with an auto-derived JSON schema."""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, get_type_hints

from medagent.tools.base import BaseTool, ToolResult

_TYPE_TO_JSON_SCHEMA: dict[type, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
}


def _schema_for_type(annotation: Any) -> dict[str, Any]:
    return _TYPE_TO_JSON_SCHEMA.get(annotation, {"type": "string"})


def _build_json_schema(func: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    hints = get_type_hints(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue
        properties[param_name] = _schema_for_type(hints.get(param_name, str))
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {"type": "object", "properties": properties, "required": required}


class FunctionTool(BaseTool):
    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._func = func

    async def run(self, **kwargs: Any) -> ToolResult:
        try:
            data = await self._func(**kwargs)
        except Exception as exc:  # noqa: BLE001 - boundary: convert to ToolResult
            return ToolResult(tool_name=self.name, success=False, error=str(exc))
        return ToolResult(tool_name=self.name, success=True, data=data)


def tool(
    name: str | None = None, description: str | None = None
) -> Callable[[Callable[..., Awaitable[Any]]], FunctionTool]:
    def decorator(func: Callable[..., Awaitable[Any]]) -> FunctionTool:
        tool_name = name or func.__name__
        tool_description = description or (inspect.getdoc(func) or "").strip()
        parameters = _build_json_schema(func)
        return FunctionTool(func, tool_name, tool_description, parameters)

    return decorator
