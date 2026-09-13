from datetime import UTC, datetime

from medagent.core.models import (
    AgentResult,
    ClinicalEvidence,
    DrugInteraction,
    PatientContext,
)
from medagent.core.types import AgentRole, InteractionSeverity
from medagent.tools.custom.report_generator import ReportGeneratorTool


async def test_report_includes_all_sections_and_citations() -> None:
    patient = PatientContext(id="P-TEST-001", name="Patient Alpha", age=60, sex="M")
    results = [
        AgentResult(
            role=AgentRole.DRUG_SAFETY,
            summary="Moderate hypoglycemia risk.",
            interactions=[
                DrugInteraction(
                    drug_a="Metformin",
                    drug_b="Glimepiride",
                    severity=InteractionSeverity.MODERATE,
                    description="Increased hypoglycemia risk.",
                    source="med-research-mcp-suite",
                    checked_at=datetime.now(UTC),
                )
            ],
        ),
        AgentResult(
            role=AgentRole.EVIDENCE,
            summary="Metformin remains first-line therapy.",
            evidence=[
                ClinicalEvidence(title="ADA Standards of Care", summary="...", source="ada-2024")
            ],
        ),
    ]

    tool = ReportGeneratorTool()
    result = await tool.run(patient, results)

    assert result.success is True
    report = result.data
    assert "Patient Alpha" in report
    assert "## Drug Safety" in report
    assert "Metformin + Glimepiride" in report
    assert "## Evidence" in report
    assert "ADA Standards of Care" in report
