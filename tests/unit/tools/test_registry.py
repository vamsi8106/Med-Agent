import pytest

from medagent.core.exceptions import ToolError
from medagent.tools.decorators import tool
from medagent.tools.registry import ToolRegistry


@tool()
async def ping() -> str:
    """Ping."""
    return "pong"


def test_registry_register_and_get() -> None:
    registry = ToolRegistry()
    registry.register(ping)
    assert registry.get("ping") is ping


def test_registry_get_missing_raises_tool_error() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolError):
        registry.get("missing")


def test_registry_schemas() -> None:
    registry = ToolRegistry()
    registry.register(ping)
    schemas = registry.schemas()
    assert schemas[0] == {
        "type": "function",
        "function": {"name": "ping", "description": "Ping.", "parameters": ping.parameters},
    }
