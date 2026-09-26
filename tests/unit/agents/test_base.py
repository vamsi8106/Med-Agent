import asyncio
import time

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
    tool_message = next(m for m in provider.histories[1] if m.role == "tool")
    assert tool_message.content.startswith("Error:")


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
    tool_message = next(m for m in provider.histories[1] if m.role == "tool")
    assert tool_message.content.startswith("Error:")


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


# --- loop engineering: deadline, duplicates, stall, parallelism -------------------


def _calls_turn(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(content="", model="mock", tool_calls=list(calls))


def _call(call_id: str, tool_name: str = "echo_tool", **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, tool_name=tool_name, arguments=arguments)


class _SlowProvider(BaseLLMProvider):
    def __init__(self, delay: float) -> None:
        self._delay = delay

    async def complete(self, messages: list[Message], tools: list | None = None) -> LLMResponse:
        await asyncio.sleep(self._delay)
        return LLMResponse(content="late answer", model="mock")


async def test_whole_loop_deadline_aborts_a_hung_provider_quickly() -> None:
    """Per-call timeouts let a hung provider (30s x 3 retries) hold the loop for
    minutes; the loop as a whole has to answer to a deadline."""
    agent = ReActAgent(
        _SlowProvider(5), ToolRegistry(), SessionMemory(), system_prompt="t", timeout_seconds=0.1
    )

    started = time.monotonic()
    with pytest.raises(ToolError, match="time limit"):
        await agent.run(_patient(), "go")

    assert time.monotonic() - started < 1.0


async def test_no_deadline_when_none_is_set() -> None:
    agent = ReActAgent(_SlowProvider(0.05), ToolRegistry(), SessionMemory(), system_prompt="t")
    assert await agent.run(_patient(), "go") == "late answer"


async def test_a_timeout_from_inside_the_loop_is_not_mistaken_for_the_deadline() -> None:
    class _RaisesTimeout(BaseLLMProvider):
        async def complete(self, messages: list[Message], tools: list | None = None) -> LLMResponse:
            raise TimeoutError("some unrelated timeout")

    agent = ReActAgent(
        _RaisesTimeout(), ToolRegistry(), SessionMemory(), system_prompt="t", timeout_seconds=30
    )

    with pytest.raises(TimeoutError, match="unrelated"):
        await agent.run(_patient(), "go")


def _counting_registry() -> tuple[ToolRegistry, list[str]]:
    executed: list[str] = []

    @tool()
    async def search(query: str) -> str:
        """Runs a search."""
        executed.append(query)
        return f"results for {query}"

    registry = ToolRegistry()
    registry.register(search)
    return registry, executed


async def test_an_identical_call_is_not_executed_twice() -> None:
    registry, executed = _counting_registry()
    provider = _RecordingScriptedProvider(
        [
            _calls_turn(_call("c1", "search", query="metformin ckd")),
            _calls_turn(_call("c2", "search", query="metformin ckd")),
            LLMResponse(content="answer", model="mock"),
        ]
    )
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="t")

    result = await agent.run_detailed(_patient(), "go")

    assert executed == ["metformin ckd"]
    assert [c.tool_name for c in result.tool_calls] == ["search"]
    third_call_tool_messages = [m for m in provider.histories[2] if m.role == "tool"]
    assert third_call_tool_messages[1].content.startswith("Error: you already made this exact call")


async def test_duplicates_ignore_case_spacing_and_argument_order() -> None:
    registry, executed = _counting_registry()
    provider = _ScriptedProvider(
        [
            _calls_turn(_call("c1", "search", query="Metformin   CKD")),
            _calls_turn(_call("c2", "search", query=" metformin ckd ")),
            _calls_turn(_call("c3", "search", query="metformin dosing")),
            LLMResponse(content="answer", model="mock"),
        ]
    )
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="t", max_iterations=6)

    await agent.run(_patient(), "go")

    assert executed == ["Metformin   CKD", "metformin dosing"]


async def test_the_same_call_twice_in_one_turn_runs_once() -> None:
    registry, executed = _counting_registry()
    provider = _RecordingScriptedProvider(
        [
            _calls_turn(_call("c1", "search", query="same"), _call("c2", "search", query="same")),
            LLMResponse(content="answer", model="mock"),
        ]
    )
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="t")

    await agent.run(_patient(), "go")

    assert executed == ["same"]
    tool_messages = [m for m in provider.histories[1] if m.role == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[1].content.startswith("Error: you already made")


async def test_a_stalled_turn_asks_for_the_answer_immediately_not_on_the_last_turn() -> None:
    registry, _ = _counting_registry()
    provider = _LastTurnProvider(
        [
            _calls_turn(_call("c1", "search", query="q")),
            _calls_turn(_call("c2", "search", query="q")),  # a repeat: no progress
            LLMResponse(content="answer", model="mock"),
        ]
    )
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="t", max_iterations=6)

    await agent.run(_patient(), "go")

    asked = [
        any("final answer now" in m.content for m in messages) for messages, _ in provider.calls
    ]
    assert asked == [False, False, True]


async def test_productive_turns_do_not_trigger_the_early_directive() -> None:
    registry, _ = _counting_registry()
    provider = _LastTurnProvider(
        [
            _calls_turn(_call("c1", "search", query="one")),
            _calls_turn(_call("c2", "search", query="two")),
            LLMResponse(content="answer", model="mock"),
        ]
    )
    agent = ReActAgent(provider, registry, SessionMemory(), system_prompt="t", max_iterations=6)

    await agent.run(_patient(), "go")

    assert not any(
        "final answer now" in m.content for messages, _ in provider.calls for m in messages
    )


def _timed_registry() -> ToolRegistry:
    @tool()
    async def wait(label: str, delay: float) -> str:
        """Waits then returns its label."""
        await asyncio.sleep(delay)
        return label

    @tool()
    async def broken(label: str) -> str:
        """Always fails."""
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(wait)
    registry.register(broken)
    return registry


async def test_a_turns_tool_calls_run_concurrently() -> None:
    provider = _RecordingScriptedProvider(
        [
            _calls_turn(
                _call("c1", "wait", label="a", delay=0.3),
                _call("c2", "wait", label="b", delay=0.3),
                _call("c3", "wait", label="c", delay=0.3),
            ),
            LLMResponse(content="answer", model="mock"),
        ]
    )
    agent = ReActAgent(provider, _timed_registry(), SessionMemory(), system_prompt="t")

    started = time.monotonic()
    await agent.run(_patient(), "go")

    assert time.monotonic() - started < 0.7  # sequential would be >= 0.9


async def test_results_keep_call_order_even_when_a_later_call_finishes_first() -> None:
    provider = _RecordingScriptedProvider(
        [
            _calls_turn(
                _call("slow", "wait", label="first", delay=0.2),
                _call("fast", "wait", label="second", delay=0.01),
            ),
            LLMResponse(content="answer", model="mock"),
        ]
    )
    agent = ReActAgent(provider, _timed_registry(), SessionMemory(), system_prompt="t")

    await agent.run(_patient(), "go")

    tool_messages = [m for m in provider.histories[1] if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["slow", "fast"]
    assert "first" in tool_messages[0].content
    assert "second" in tool_messages[1].content


async def test_one_failing_tool_does_not_affect_its_siblings() -> None:
    provider = _RecordingScriptedProvider(
        [
            _calls_turn(
                _call("c1", "broken", label="x"), _call("c2", "wait", label="fine", delay=0.01)
            ),
            LLMResponse(content="answer", model="mock"),
        ]
    )
    agent = ReActAgent(provider, _timed_registry(), SessionMemory(), system_prompt="t")

    await agent.run(_patient(), "go")

    tool_messages = [m for m in provider.histories[1] if m.role == "tool"]
    assert tool_messages[0].content.startswith("Error:")
    assert "fine" in tool_messages[1].content


# --- metrics ------------------------------------------------------------------------------

from tests.conftest import metric_value  # noqa: E402


async def test_tool_calls_are_counted_by_outcome() -> None:
    def total(outcome: str, tool_name: str = "search") -> float:
        return metric_value("medagent_tool_calls_total", tool=tool_name, outcome=outcome)

    before = {o: total(o) for o in ("success", "duplicate")}
    registry, _ = _counting_registry()
    provider = _ScriptedProvider(
        [
            _calls_turn(_call("c1", "search", query="q")),
            _calls_turn(_call("c2", "search", query="q")),
            LLMResponse(content="answer", model="mock"),
        ]
    )

    await ReActAgent(provider, registry, SessionMemory(), system_prompt="t").run(_patient(), "go")

    assert total("success") - before["success"] == 1
    assert total("duplicate") - before["duplicate"] == 1


async def test_a_model_invented_tool_name_never_becomes_a_metric_label() -> None:
    """The model chooses tool names; an unbounded label would let it (or an
    injected prompt) blow up metric cardinality."""
    before = metric_value("medagent_tool_calls_total", tool="unknown", outcome="error")
    provider = _ScriptedProvider(
        [
            _calls_turn(_call("c1", "definitely_not_a_real_tool_xyz")),
            LLMResponse(content="answer", model="mock"),
        ]
    )

    await ReActAgent(provider, ToolRegistry(), SessionMemory(), system_prompt="t").run(
        _patient(), "go"
    )

    assert metric_value("medagent_tool_calls_total", tool="unknown", outcome="error") - before == 1
    assert (
        metric_value(
            "medagent_tool_calls_total", tool="definitely_not_a_real_tool_xyz", outcome="error"
        )
        == 0
    )


async def test_the_loop_deadline_raises_a_distinct_timeout_error() -> None:
    from medagent.agents.base import ReActTimeoutError

    agent = ReActAgent(
        _SlowProvider(5), ToolRegistry(), SessionMemory(), system_prompt="t", timeout_seconds=0.05
    )
    with pytest.raises(ReActTimeoutError):
        await agent.run(_patient(), "go")
