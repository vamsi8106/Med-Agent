"""`/drug-check`: "can I add drug X?" -- the patient's saved medications plus the
new drug, every pair checked by Drug Safety directly (no triage)."""

from medagent.core.models import Medication, PatientContext
from medagent.core.types import AgentRole, InteractionSeverity
from tests.eval.agent_cases.base import AgentCase

_DRUG_SAFETY = frozenset({AgentRole.DRUG_SAFETY})

CASES: list[AgentCase] = [
    AgentCase(
        id="dc-01",
        category="drug_check",
        workflow="drug_check",
        patient=PatientContext(
            id="P-EVAL-951",
            name="Patient Pi Two",
            age=72,
            sex="F",
            medications=[Medication(name="Warfarin"), Medication(name="Metformin")],
        ),
        new_drug="Aspirin",
        message="Can I add Aspirin?",
        source="FDA warfarin label (bleeding risk with antiplatelet agents)",
        expected_roles=_DRUG_SAFETY,
        drug_pair=("Warfarin", "Aspirin"),
        min_severity=InteractionSeverity.MODERATE,
        clinical_severity=InteractionSeverity.MAJOR,
    ),
    AgentCase(
        id="dc-02",
        category="drug_check",
        workflow="drug_check",
        patient=PatientContext(
            id="P-EVAL-952",
            name="Patient Rho Two",
            age=59,
            sex="M",
            allergies=["Ibuprofen"],
            medications=[Medication(name="Lisinopril")],
        ),
        new_drug="Ibuprofen",
        message="Can I add Ibuprofen?",
        source="Recorded allergy matches the drug being added",
        expected_roles=_DRUG_SAFETY,
        drug_pair=("Lisinopril", "Ibuprofen"),
        expect_allergy_flag=True,
    ),
    AgentCase(
        id="dc-03",
        category="drug_check",
        workflow="drug_check",
        patient=PatientContext(
            id="P-EVAL-953",
            name="Patient Sigma Two",
            age=63,
            sex="F",
            medications=[Medication(name="Simvastatin")],
        ),
        new_drug="Clarithromycin",
        message="Can I add Clarithromycin?",
        source="FDA simvastatin label: strong CYP3A4 inhibitors incl. clarithromycin "
        "contraindicated",
        expected_roles=_DRUG_SAFETY,
        drug_pair=("Simvastatin", "Clarithromycin"),
        min_severity=InteractionSeverity.MODERATE,
        clinical_severity=InteractionSeverity.CONTRAINDICATED,
    ),
]
