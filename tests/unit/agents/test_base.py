import pytest

from medagent.agents.base import ReActAgent
from medagent.core.exceptions import ToolError
from medagent.core.interfaces import BaseLLMProvider
from medagent.core.models import LLMResponse, Message, PatientContext, ToolCall
from medagent.memory.session import SessionMemory
from medagent.tools.decorators import tool
from medagent.tools.registry import ToolRegistry


class _ScriptedProvider(BaseLLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    async def complete(self, messages: list[Message], tools: list | None = None) -> LLMResponse:
        return self._responses.pop(0)


@tool()
async def echo_tool(value: str) -> str:
    """Echoes the given value."""
    return value


def _patient() -> PatientContext:
    return PatientContext(id="P-TEST-001", name="Patient Alpha", age=40, sex="F")


async def test_react_agent_returns_final_answer_without_tool_call() -> None:
    provider = _ScriptedProvider([LLMResponse(content="final answer", model="mock")])
    registry = ToolRegistry()
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="test")

    result = await agent.run(_patient(), "hello")

    assert result == "final answer"


async def test_react_agent_executes_tool_call_then_returns_final_answer() -> None:
    provider = _ScriptedProvider(
        [
            LLMResponse(
                content="",
                model="mock",
                tool_calls=[ToolCall(tool_name="echo_tool", arguments={"value": "hi"})],
            ),
            LLMResponse(content="final answer", model="mock"),
        ]
    )
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="test")

    result = await agent.run(_patient(), "please echo")

    assert result == "final answer"


async def test_react_agent_raises_when_max_iterations_exceeded() -> None:
    looping_call = ToolCall(tool_name="echo_tool", arguments={"value": "hi"})
    provider = _ScriptedProvider(
        [LLMResponse(content="", model="mock", tool_calls=[looping_call])] * 3
    )
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="test", max_iterations=3)

    with pytest.raises(ToolError):
        await agent.run(_patient(), "please echo")
