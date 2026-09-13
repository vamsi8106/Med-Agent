"""Tool registry: holds tool instances and resolves them by name for agents."""

from medagent.core.exceptions import ToolError
from medagent.core.interfaces import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"No tool registered under name: {name}") from exc

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, object]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": getattr(t, "parameters", {}),
            }
            for t in self._tools.values()
        ]
