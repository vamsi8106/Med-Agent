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


class _RecordingScriptedProvider(_ScriptedProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(responses)
        self.histories: list[list[Message]] = []

    async def complete(self, messages: list[Message], tools: list | None = None) -> LLMResponse:
        self.histories.append(list(messages))
        return await super().complete(messages, tools)


def _tool_turn(call_id: str, **arguments: str) -> LLMResponse:
    return LLMResponse(
        content="",
        model="mock",
        tool_calls=[ToolCall(id=call_id, tool_name="echo_tool", arguments=arguments)],
        usage={"prompt_tokens": 10, "completion_tokens": 2},
    )


async def test_assistant_tool_call_turn_precedes_its_result_with_matching_id() -> None:
    """Real providers reject a tool message with no preceding assistant turn
    that requested it; the loop previously never recorded that turn."""
    provider = _RecordingScriptedProvider(
        [_tool_turn("call_1", value="hi"), LLMResponse(content="done", model="mock")]
    )
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="test")

    await agent.run(_patient(), "please echo")

    second_call = provider.histories[1]
    roles = [m.role for m in second_call]
    assert roles == ["system", "user", "assistant", "tool"]
    assert second_call[2].tool_calls is not None
    assert second_call[2].tool_calls[0].id == "call_1"
    assert second_call[3].tool_call_id == "call_1"


async def test_unknown_tool_is_reported_to_the_model_not_raised() -> None:
    provider = _RecordingScriptedProvider(
        [
            LLMResponse(
                content="",
                model="mock",
                tool_calls=[ToolCall(id="c1", tool_name="no_such_tool", arguments={})],
            ),
            LLMResponse(content="recovered", model="mock"),
        ]
    )
    agent = ReActAgent(provider, ToolRegistry(), SessionMemory(), system_prompt="test")

    result = await agent.run(_patient(), "go")

    assert result == "recovered"
    assert provider.histories[1][-1].content.startswith("Error:")


async def test_tool_failure_is_returned_to_the_model_as_an_error_result() -> None:
    provider = _RecordingScriptedProvider(
        [
            LLMResponse(
                content="",
                model="mock",
                tool_calls=[ToolCall(id="c1", tool_name="echo_tool", arguments={"wrong": "x"})],
            ),
            LLMResponse(content="recovered", model="mock"),
        ]
    )
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="test")

    result = await agent.run(_patient(), "go")

    assert result == "recovered"
    assert provider.histories[1][-1].content.startswith("Error:")


async def test_tool_output_is_delimited_as_untrusted_and_capped() -> None:
    @tool()
    async def big_tool(value: str) -> str:
        """Returns a huge payload."""
        return "word " * 5000

    provider = _RecordingScriptedProvider(
        [
            LLMResponse(
                content="",
                model="mock",
                tool_calls=[ToolCall(id="c1", tool_name="big_tool", arguments={"value": "x"})],
            ),
            LLMResponse(content="done", model="mock"),
        ]
    )
    registry = ToolRegistry()
    registry.register(big_tool)
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="test")

    await agent.run(_patient(), "go")

    tool_message = provider.histories[1][-1]
    assert "<<<" in tool_message.content
    assert "truncated" in tool_message.content
    assert len(tool_message.content) < 6000


async def test_run_detailed_reports_tool_calls_usage_and_iterations() -> None:
    provider = _ScriptedProvider(
        [
            _tool_turn("c1", value="a"),
            _tool_turn("c2", value="b"),
            LLMResponse(content="done", model="mock", usage={"prompt_tokens": 7}),
        ]
    )
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="test")

    result = await agent.run_detailed(_patient(), "go")

    assert result.answer == "done"
    assert result.iterations == 3
    assert [c.tool_name for c in result.tool_calls] == ["echo_tool", "echo_tool"]
    assert result.usage == {"prompt_tokens": 27, "completion_tokens": 4}


async def test_prompt_budget_stops_the_loop_before_an_oversized_llm_call() -> None:
    @tool()
    async def big_tool(value: str) -> str:
        """Returns a large payload."""
        return "word " * 400

    provider = _ScriptedProvider(
        [
            LLMResponse(
                content="",
                model="mock",
                tool_calls=[ToolCall(id="c1", tool_name="big_tool", arguments={"value": "x"})],
            ),
            LLMResponse(content="never reached", model="mock"),
        ]
    )
    registry = ToolRegistry()
    registry.register(big_tool)
    agent = ReActAgent(
        provider, registry, SessionMemory(), system_prompt="test", max_prompt_tokens=100
    )

    with pytest.raises(ToolError, match="budget"):
        await agent.run(_patient(), "go")


class _LastTurnProvider(_ScriptedProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(responses)
        self.calls: list[tuple[list[Message], bool]] = []

    async def complete(self, messages: list[Message], tools: list | None = None) -> LLMResponse:
        self.calls.append((list(messages), bool(tools)))
        return await super().complete(messages, tools)


async def test_last_turn_tells_the_model_to_answer_but_keeps_tools_attached() -> None:
    """A model that keeps searching used to burn every iteration and abort the
    run. On the final allowed turn a directive asks for the answer; tools stay
    attached because omitting them makes Groq 400 if the model calls one anyway."""
    provider = _LastTurnProvider(
        [
            _tool_turn("c1", value="a"),
            _tool_turn("c2", value="b"),
            LLMResponse(content="answer", model="m"),
        ]
    )
    registry = ToolRegistry()
    registry.register(echo_tool)
    session = SessionMemory()
    agent = ReActAgent(provider, registry, session, system_prompt="t", max_iterations=3)

    result = await agent.run(_patient(), "go")

    assert result == "answer"
    assert all(has_tools for _, has_tools in provider.calls)
    directive_seen = [
        any("final answer now" in m.content for m in messages) for messages, _ in provider.calls
    ]
    assert directive_seen == [False, False, True]
    assert not any("final answer now" in m.content for m in session.get_messages())


async def test_no_final_answer_directive_when_no_tool_has_run_yet() -> None:
    provider = _LastTurnProvider([LLMResponse(content="answer", model="m")])
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="t", max_iterations=1)

    await agent.run(_patient(), "go")

    assert not any("final answer now" in m.content for m in provider.calls[0][0])
