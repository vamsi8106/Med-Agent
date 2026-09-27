from datetime import UTC, datetime

from medagent.core.models import (
    AgentResult,
    ClinicalEvidence,
    DrugInteraction,
    LabChange,
    LabResult,
    MedicationChange,
    PatientContext,
    RecordChanges,
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


def _changes_patient(changes: object) -> PatientContext:
    return PatientContext(
        id="P-TEST-001",
        name="Patient Alpha",
        age=60,
        sex="M",
        changes_since_last_visit=changes,  # type: ignore[arg-type]
    )


async def test_report_lists_changes_since_last_visit() -> None:
    changes = RecordChanges(
        since=datetime(2026, 1, 15, tzinfo=UTC),
        medications_started=["Lisinopril"],
        medications_changed=[MedicationChange(name="Metformin", before="500 mg", after="1000 mg")],
        lab_changes=[
            LabChange(
                test_name="HbA1c",
                unit="%",
                previous_value=8.1,
                current_value=9.8,
                reference_low=4.0,
                reference_high=5.7,
                collected_at=datetime(2026, 6, 1, tzinfo=UTC),
            )
        ],
    )

    report = (await ReportGeneratorTool().run(_changes_patient(changes), [])).data

    assert "## Changes Since Last Visit (2026-01-15)" in report
    assert "- Started: Lisinopril" in report
    assert "- Changed: Metformin: 500 mg -> 1000 mg" in report
    assert "- HbA1c: 8.1 -> 9.8 % (reference 4.0-5.7), collected 2026-06-01" in report


async def test_report_says_so_when_nothing_changed() -> None:
    changes = RecordChanges(since=datetime(2026, 1, 15, tzinfo=UTC))

    report = (await ReportGeneratorTool().run(_changes_patient(changes), [])).data

    assert "No recorded changes to medications, conditions, allergies or labs." in report


async def test_report_has_no_changes_section_without_a_baseline() -> None:
    report = (await ReportGeneratorTool().run(_changes_patient(None), [])).data

    assert "Changes Since Last Visit" not in report
