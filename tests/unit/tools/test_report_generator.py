from datetime import UTC, datetime

from medagent.core.models import (
    AgentResult,
    ClinicalEvidence,
    DrugInteraction,
    LabResult,
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
    assert "## Known Allergies" not in report
    assert "## Lab Flags" not in report


async def test_report_includes_allergies_section_when_present() -> None:
    patient = PatientContext(
        id="P-TEST-002", name="Patient Beta", age=45, sex="F", allergies=["Penicillin"]
    )

    tool = ReportGeneratorTool()
    result = await tool.run(patient, [])

    assert "## Known Allergies" in result.data
    assert "Penicillin" in result.data


async def test_report_lab_flags_section_shows_only_abnormal_results() -> None:
    patient = PatientContext(
        id="P-TEST-003",
        name="Patient Gamma",
        age=55,
        sex="M",
        lab_results=[
            LabResult(
                test_name="HbA1c",
                value=9.5,
                unit="%",
                reference_low=4.0,
                reference_high=5.7,
                collected_at=datetime.now(UTC),
            ),
            LabResult(
                test_name="Sodium",
                value=140,
                unit="mmol/L",
                reference_low=135,
                reference_high=145,
                collected_at=datetime.now(UTC),
            ),
        ],
    )

    tool = ReportGeneratorTool()
    result = await tool.run(patient, [])

    assert "## Lab Flags" in result.data
    assert "HbA1c" in result.data
    assert "Sodium" not in result.data
