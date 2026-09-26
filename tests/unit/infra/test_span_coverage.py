"""One assessment should read as a tree of meaningful spans -- and none of them
may carry patient data."""

from unittest.mock import AsyncMock

import httpx
import pytest
from opentelemetry import trace

from medagent.agents.base import ReActAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.exceptions import ProviderError
from medagent.core.interfaces import BaseLLMProvider
from medagent.core.models import (
    AgentResult,
    LLMResponse,
    Medication,
    Message,
    PatientContext,
    ToolCall,
)
from medagent.core.types import AgentRole
from medagent.memory.session import SessionMemory
from medagent.tools.decorators import tool
from medagent.tools.mcp.http_base import HttpMCPClient
from medagent.tools.registry import ToolRegistry
from medagent.workflows.patient_assessment import run_patient_assessment
from tests.conftest import FakeClock, span_text
from tests.unit.test_groq_provider import _FakeSDK, _ok_response, _provider, _status_error


def _by_name(spans, name: str):
    return [s for s in spans.get_finished_spans() if s.name == name]


# --- workflow nodes ------------------------------------------------------------------------


def _patient() -> PatientContext:
    return PatientContext(
        id="P-TEST-1",
        name="Patient Alpha",
        age=60,
        sex="F",
        medications=[Medication(name="Metformin"), Medication(name="Glimepiride")],
    )


async def test_an_assessment_produces_a_span_per_workflow_node(spans) -> None:
    drug_safety = AsyncMock()
    drug_safety.run_result.return_value = AgentResult(role=AgentRole.DRUG_SAFETY, summary="ok")
    evidence = AsyncMock()
    evidence.gather_evidence.return_value = AgentResult(role=AgentRole.EVIDENCE, summary="ok")

    with trace.get_tracer("t").start_as_current_span("request") as root:
        await run_patient_assessment(
            TriageAgent(),
            drug_safety,
            evidence,
            AsyncMock(),
            ReportAgent(),
            _patient(),
            "Check interactions for current medications and any relevant treatment evidence",
        )

    names = {s.name for s in spans.get_finished_spans()}
    assert {"node.init_run", "node.triage", "node.drug_safety", "node.evidence", "node.report"} <= (
        names
    )
    # all in the request's trace: the span context crossed into LangGraph's node tasks
    trace_id = root.get_span_context().trace_id
    assert all(s.context.trace_id == trace_id for s in spans.get_finished_spans())
    assert _by_name(spans, "node.evidence")[0].attributes["medagent.outcome"] == "completed"


async def test_a_failed_specialist_span_says_why_without_the_error_text(spans) -> None:
    drug_safety = AsyncMock()
    drug_safety.run_result.return_value = AgentResult(role=AgentRole.DRUG_SAFETY, summary="ok")
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = ProviderError(
        "org_01SECRET429", reason="rate_limited", retryable=True, retry_after=5
    )

    await run_patient_assessment(
        TriageAgent(),
        drug_safety,
        evidence,
        AsyncMock(),
        ReportAgent(),
        _patient(),
        "Check interactions for current medications and any relevant treatment evidence",
    )

    span = _by_name(spans, "node.evidence")[0]
    assert span.attributes["medagent.outcome"] == "failed"
    assert span.attributes["medagent.failure_reason"] == "rate_limited"
    assert "org_01secret429" not in span_text(span)


# --- ReAct loop ----------------------------------------------------------------------------


class _Scripted(BaseLLMProvider):
    def __init__(self, *turns: LLMResponse) -> None:
        self._turns = list(turns)

    async def complete(self, messages: list[Message], tools: list | None = None) -> LLMResponse:
        return self._turns.pop(0)


def _search_tool() -> ToolRegistry:
    @tool()
    async def search(query: str) -> str:
        """Searches."""
        return f"results for {query}"

    registry = ToolRegistry()
    registry.register(search)
    return registry


def _call(call_id: str, query: str) -> LLMResponse:
    return LLMResponse(
        content="",
        model="m",
        tool_calls=[ToolCall(id=call_id, tool_name="search", arguments={"query": query})],
    )


async def test_react_turns_and_tool_calls_are_spans_with_counts_not_content(spans) -> None:
    agent = ReActAgent(
        _Scripted(_call("c1", "zzsecretquery"), LLMResponse(content="the answer", model="m")),
        _search_tool(),
        SessionMemory(),
        system_prompt="t",
    )

    await agent.run(_patient(), "zzsecretquestion")

    turns = sorted(_by_name(spans, "react.turn"), key=lambda s: s.attributes["react.iteration"])
    assert [t.attributes["react.tool_call_count"] for t in turns] == [1, 0]
    tool_span = _by_name(spans, "react.tool")[0]
    assert tool_span.attributes["react.tool"] == "search"
    assert tool_span.attributes["react.outcome"] == "success"
    everything = " ".join(span_text(s) for s in spans.get_finished_spans())
    assert "zzsecretquery" not in everything
    assert "zzsecretquestion" not in everything
    assert "the answer" not in everything


async def test_a_model_invented_tool_name_is_labelled_unknown_on_its_span(spans) -> None:
    agent = ReActAgent(
        _Scripted(
            LLMResponse(
                content="",
                model="m",
                tool_calls=[ToolCall(id="c1", tool_name="drop_all_tables_zz", arguments={})],
            ),
            LLMResponse(content="answer", model="m"),
        ),
        _search_tool(),
        SessionMemory(),
        system_prompt="t",
    )

    await agent.run(_patient(), "go")

    span = _by_name(spans, "react.tool")[0]
    assert span.attributes["react.tool"] == "unknown"
    assert span.attributes["react.outcome"] == "error"


async def test_the_forced_answer_turn_is_flagged(spans) -> None:
    agent = ReActAgent(
        _Scripted(_call("c1", "q"), _call("c2", "q"), LLMResponse(content="answer", model="m")),
        _search_tool(),
        SessionMemory(),
        system_prompt="t",
        max_iterations=6,
    )

    await agent.run(_patient(), "go")

    forced = {
        t.attributes["react.iteration"]: t.attributes["react.forced_answer"]
        for t in _by_name(spans, "react.turn")
    }
    assert forced == {1: False, 2: False, 3: True}  # the repeat stalled, so turn 3 must answer


# --- LLM call ------------------------------------------------------------------------------


async def test_the_llm_span_has_model_tokens_and_outcome_but_no_prompt_text(
    spans, clock: FakeClock
) -> None:
    provider, _ = _provider(
        _FakeSDK(_ok_response()),
    )

    await provider.complete([Message(role="user", content="PROMPT-WITH-PATIENT-DATA-ZZ")])

    span = _by_name(spans, "llm.chat")[0]
    assert span.attributes["gen_ai.system"] == "groq"
    assert span.attributes["gen_ai.request.model"] == "test-model"
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 5
    assert span.attributes["medagent.llm.outcome"] == "success"
    assert "patient-data-zz" not in span_text(span)


async def test_a_failed_llm_call_span_carries_the_outcome_and_an_error_status(
    spans, clock: FakeClock
) -> None:
    provider, _ = _provider(_FakeSDK(_status_error(404, message="model does not exist")))

    with pytest.raises(ProviderError):
        await provider.complete([Message(role="user", content="hi")])

    span = _by_name(spans, "llm.chat")[0]
    assert span.attributes["medagent.llm.outcome"] == "client_error"
    assert span.status.status_code.name == "ERROR"


async def test_retries_are_span_events_so_a_slow_call_explains_itself(
    spans, clock: FakeClock
) -> None:
    provider, _ = _provider(_FakeSDK(_status_error(500), _status_error(500), _ok_response()))

    await provider.complete([Message(role="user", content="hi")])

    events = [e for e in _by_name(spans, "llm.chat")[0].events if e.name == "retry"]
    assert [e.attributes["medagent.retry.attempt"] for e in events] == [1, 2]
    assert events[0].attributes["medagent.retry.reason"] == "server_error"
    assert events[0].attributes["medagent.retry.target"] == "llm"


async def test_a_breaker_refusal_is_a_span_event(spans, clock: FakeClock) -> None:
    provider, _ = _provider(_FakeSDK(_status_error(500)))
    for _ in range(5):
        with pytest.raises(ProviderError):
            await provider.complete([Message(role="user", content="hi")])
    spans.clear()

    with pytest.raises(ProviderError):
        await provider.complete([Message(role="user", content="hi")])

    span = _by_name(spans, "llm.chat")[0]
    assert any(e.name == "circuit_open" for e in span.events)
    assert span.attributes["medagent.llm.outcome"] == "circuit_open"


# --- propagation to the MCP servers --------------------------------------------------------


async def test_the_trace_is_propagated_to_the_mcp_server_via_traceparent(spans) -> None:
    received: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received.update(request.headers)
        return httpx.Response(200, json={"ok": True})

    client = HttpMCPClient("http://test-server", name="medical")
    client._client = httpx.AsyncClient(
        base_url="http://test-server", transport=httpx.MockTransport(handler)
    )

    with trace.get_tracer("t").start_as_current_span("request") as root:
        await client.request("POST", "/call-tool", json={"name": "search-drugs"})

    header = received["traceparent"]
    version, trace_id, _span_id, _flags = header.split("-")
    assert (version, trace_id) == ("00", format(root.get_span_context().trace_id, "032x"))
    mcp_span = _by_name(spans, "mcp_http.post./call-tool")[0]
    assert mcp_span.attributes["medagent.mcp.server"] == "medical"
    assert mcp_span.attributes["medagent.mcp.outcome"] == "success"
    await client._client.aclose()


async def test_a_callers_own_headers_survive_trace_injection() -> None:
    received: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received.update(request.headers)
        return httpx.Response(200, json={})

    client = HttpMCPClient("http://test-server")
    client._client = httpx.AsyncClient(
        base_url="http://test-server", transport=httpx.MockTransport(handler)
    )

    await client.request("GET", "/x", headers={"X-Custom": "keep-me"})

    assert received["x-custom"] == "keep-me"
    await client._client.aclose()
