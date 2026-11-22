"""How the agent eval watches a run without changing `src/`.

- `RecordingLLM`: every LLM call (tools offered, tool calls returned, tokens,
  the error reason and Retry-After when it fails);
- `observe_specialists`: records each specialist's `AgentResult` and triage's
  routing as they return, the same way in every workflow (follow-up and
  drug-check don't hand their per-agent results back);
- spy MCP clients: every query sent to an external API (identifier-leak
  check); for injection cases, hostile text added to what a server returned;
  for degraded cases, a server made to fail;
- spans (an in-memory exporter on the tracer provider) and the
  `medagent_evidence_react_total` counter.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY

from medagent.api.state import Specialists
from medagent.core.config import Settings
from medagent.core.exceptions import MCPError, ProviderError
from medagent.core.interfaces import BaseLLMProvider
from medagent.core.models import AgentResult, LLMResponse, Message, PatientContext
from medagent.core.types import AgentRole
from medagent.tools.mcp.healthcare import HealthcareMCPClient
from medagent.tools.mcp.medical import MedicalMCPClient
from medagent.tools.mcp.research import ResearchMCPClient
from tests.eval.agent_cases import Injection

REACT_PATHS = ("completed", "fallback_ungrounded", "fallback_timeout", "fallback_error")

# Which spied server a degraded case breaks for each specialist.
FAILING_SERVER: dict[AgentRole, str] = {
    AgentRole.TRIAL_FINDER: "trials",
    AgentRole.DRUG_SAFETY: "interactions",
    AgentRole.EVIDENCE: "literature",
}


# --- recording LLM ---------------------------------------------------------------------


@dataclass
class LLMCall:
    offered: list[str]
    issued: list[str]
    tokens: int
    error: str | None
    retry_after: float | None = None


class RecordingLLM(BaseLLMProvider):
    def __init__(self, inner: BaseLLMProvider) -> None:
        self.inner = inner
        self.calls: list[LLMCall] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        offered = [t.get("function", {}).get("name", "") for t in tools or []]
        try:
            response = await self.inner.complete(messages, tools)
        except ProviderError as exc:
            self.calls.append(LLMCall(offered, [], 0, exc.reason, exc.retry_after))
            raise
        tokens = sum(response.usage.values())
        issued = [call.tool_name for call in response.tool_calls]
        self.calls.append(LLMCall(offered, issued, tokens, None))
        return response


# --- per-run probe ---------------------------------------------------------------------


@dataclass
class Probe:
    """Per-run state the spies and specialist wrappers write into."""

    outbound: list[tuple[str, str]] = field(default_factory=list)  # (where, query)
    injection: Injection | None = None
    failing_server: str | None = None
    results: list[AgentResult] = field(default_factory=list)
    roles: list[AgentRole] | None = None

    def reset(self, injection: Injection | None, failing_server: str | None) -> None:
        self.outbound.clear()
        self.results.clear()
        self.roles = None
        self.injection = injection
        self.failing_server = failing_server

    def injecting(self, where: str) -> Injection | None:
        injection = self.injection
        return injection if injection is not None and injection.where == where else None

    def fail_if_broken(self, server: str) -> None:
        if self.failing_server == server:
            raise MCPError(f"{server} made to fail by the eval", reason="server_error")


# --- spy MCP clients -------------------------------------------------------------------


# The spies keep the probe as `_eval`: `probe` is HttpMCPClient's readiness
# check, which /ready calls.


class SpyMedicalClient(MedicalMCPClient):
    def __init__(self, settings: Settings, probe: Probe) -> None:
        super().__init__(settings)
        self._eval = probe

    async def search_medical_literature(self, query: str, max_results: int = 10) -> Any:
        self._eval.outbound.append(("literature", query))
        self._eval.fail_if_broken("literature")
        content = await super().search_medical_literature(query, max_results)
        if injection := self._eval.injecting("literature"):
            # In front: the agent caps long results, which would cut a suffix.
            blocks = content if isinstance(content, list) else [str(content)]
            return [{"text": injection.text}, *blocks]
        return content


class SpyHealthcareClient(HealthcareMCPClient):
    def __init__(self, settings: Settings, probe: Probe) -> None:
        super().__init__(settings)
        self._eval = probe

    async def clinical_trials_search(
        self, condition: str, status: str | None = None, max_results: int | None = None
    ) -> Any:
        self._eval.outbound.append(("trials", condition))
        self._eval.fail_if_broken("trials")
        content = await super().clinical_trials_search(condition, status, max_results)
        if injection := self._eval.injecting("trials"):
            data = content.get("data", content) if isinstance(content, dict) else content
            return {"data": f"{injection.text}\n\n{data}"}
        return content


class SpyResearchClient(ResearchMCPClient):
    def __init__(self, settings: Settings, probe: Probe) -> None:
        super().__init__(settings)
        self._eval = probe

    async def comprehensive_analysis(self, drug_name: str, condition: str) -> Any:
        self._eval.outbound.append(("interactions", f"{drug_name} {condition}"))
        self._eval.fail_if_broken("interactions")
        content = await super().comprehensive_analysis(drug_name, condition)
        if injection := self._eval.injecting("interactions"):
            # The executive summary is what InteractionCheckerTool turns into
            # the interaction description the Drug Safety prompt reads.
            data = content.get("data", content) if isinstance(content, dict) else {}
            data = dict(data) if isinstance(data, dict) else {}
            data["executiveSummary"] = f"{injection.text} {data.get('executiveSummary', '')}"
            return {"success": True, "data": data}
        return content


# --- specialist results ----------------------------------------------------------------

_Specialist = Callable[[PatientContext, str], Awaitable[AgentResult]]


def observe_specialists(specialists: Specialists, probe: Probe) -> None:
    """Wraps each specialist's entry point (and triage's routing) so their
    results land in `probe`, whatever workflow called them."""

    def recording(method: _Specialist) -> _Specialist:
        async def wrapped(context: PatientContext, message: str) -> AgentResult:
            result = await method(context, message)
            probe.results.append(result)
            return result

        return wrapped

    route = specialists.triage.route

    async def routing(context: PatientContext, message: str) -> list[AgentRole]:
        roles = await route(context, message)
        probe.roles = list(roles)
        return roles

    # Instance attributes shadow the class's methods for these objects only.
    vars(specialists.triage)["route"] = routing
    for agent, name in (
        (specialists.drug_safety, "run_result"),
        (specialists.evidence, "gather_evidence"),
        (specialists.trial_finder, "find_trials"),
    ):
        vars(agent)[name] = recording(getattr(agent, name))


# --- spans and counters ----------------------------------------------------------------


def capture_spans() -> InMemorySpanExporter:
    """Adds an in-memory export to the global tracer provider (installing one
    if none exists, as when run from the CLI)."""
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


def react_counts() -> dict[str, float]:
    return {
        path: REGISTRY.get_sample_value("medagent_evidence_react_total", {"outcome": path}) or 0.0
        for path in REACT_PATHS
    }
