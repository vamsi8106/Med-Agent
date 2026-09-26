"""Scenario tests: what the whole harness does when a dependency misbehaves.

Each drives the real GroqProvider (retry policy + circuit breaker + error
classification) through a real EvidenceAgent and the real assessment workflow,
with only the network faked. A fake clock makes waiting instant, so a policy
that would hold a request for minutes is asserted in microseconds.
"""

from unittest.mock import AsyncMock

import pytest

from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.exceptions import AllAgentsFailedError
from medagent.core.models import PatientContext
from medagent.workflows.patient_assessment import run_patient_assessment
from tests.conftest import FakeClock, metric_value
from tests.unit.test_groq_provider import _FakeSDK, _ok_response, _provider, _status_error


class _MedicalClient:
    def __init__(self) -> None:
        self.search_medical_literature = AsyncMock(
            return_value=[{"text": "Metformin lowers HbA1c."}]
        )

    async def __aenter__(self) -> "_MedicalClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def _evidence_agent(llm: object) -> EvidenceAgent:
    guidelines = AsyncMock()
    guidelines.run.return_value = []
    return EvidenceAgent(llm, _MedicalClient(), guidelines)  # type: ignore[arg-type]


async def _assess(evidence: EvidenceAgent) -> str:
    return await run_patient_assessment(
        TriageAgent(),
        AsyncMock(),
        evidence,
        AsyncMock(),
        ReportAgent(),
        PatientContext(id="P-TEST-1", name="Patient Alpha", age=60, sex="F"),
        "any updates?",  # routes to the evidence agent only
    )


async def test_a_rate_limit_storm_fails_fast_instead_of_stacking_retries(clock: FakeClock) -> None:
    """The original bug. Groq's daily cap answers 429 "try again in 8m". The SDK
    retried it 3x and our decorator wrapped that 3x more (9 HTTP calls per LLM
    call, 19-91s per request). Now: one call per LLM attempt, no waiting."""
    sdk = _FakeSDK(_status_error(429, {"retry-after": "487"}))
    provider, _ = _provider(sdk)

    with pytest.raises(AllAgentsFailedError) as excinfo:
        await _assess(_evidence_agent(provider))

    # ReAct attempt + the deterministic fallback = 2 LLM calls, 1 HTTP call each.
    assert sdk.calls == 2
    assert clock.sleeps == []
    # ...and the caller is told to back off, not that the service is down.
    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after == 487.0
    assert "rate-limited; try again in about 9 minutes" in str(excinfo.value)


async def test_a_wrong_model_name_is_attempted_once_not_nine_times(clock: FakeClock) -> None:
    """CI failed with 404 model_not_found; the old policy retried every one."""
    sdk = _FakeSDK(_status_error(404, message="The model `x` does not exist"))
    provider, _ = _provider(sdk)

    with pytest.raises(AllAgentsFailedError):
        await _assess(_evidence_agent(provider))

    assert sdk.calls == 2
    assert clock.sleeps == []


async def test_a_groq_outage_opens_the_breaker_and_later_requests_never_reach_groq(
    clock: FakeClock,
) -> None:
    sdk = _FakeSDK(_status_error(500))
    provider, _ = _provider(sdk)
    evidence = _evidence_agent(provider)

    # Hammer it until the breaker opens (each assessment makes two LLM attempts).
    for _ in range(4):
        with pytest.raises(AllAgentsFailedError):
            await _assess(evidence)
    calls_once_open = sdk.calls
    circuit_open_before = metric_value(
        "medagent_llm_calls_total", provider="groq", outcome="circuit_open"
    )

    with pytest.raises(AllAgentsFailedError) as excinfo:
        await _assess(evidence)

    assert sdk.calls == calls_once_open  # not one more request reached Groq
    assert "language model service was unavailable" in str(excinfo.value)
    circuit_open_after = metric_value(
        "medagent_llm_calls_total", provider="groq", outcome="circuit_open"
    )
    assert circuit_open_after - circuit_open_before >= 1


async def test_a_transient_blip_is_absorbed_and_the_request_still_succeeds(
    clock: FakeClock,
) -> None:
    """Two 503s then success: the doctor never sees it."""
    sdk = _FakeSDK(_status_error(500), _status_error(500), _ok_response())
    provider, _ = _provider(sdk)
    evidence = _evidence_agent(provider)

    result = await evidence.gather_evidence(
        PatientContext(id="P-TEST-1", name="Patient Alpha", age=60, sex="F"), "any updates?"
    )

    assert result.summary
    assert len(clock.sleeps) == 2
