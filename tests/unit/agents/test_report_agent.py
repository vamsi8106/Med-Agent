from medagent.agents.report_agent import ReportAgent
from medagent.core.models import AgentResult, PatientContext
from medagent.core.types import AgentRole


async def test_generate_produces_markdown_report() -> None:
    patient = PatientContext(id="P-TEST-001", name="Patient Alpha", age=60, sex="M")
    results = [AgentResult(role=AgentRole.EVIDENCE, summary="Findings here.")]

    agent = ReportAgent()
    report = await agent.generate(patient, results)

    assert "Patient Alpha" in report
    assert "Findings here." in report
