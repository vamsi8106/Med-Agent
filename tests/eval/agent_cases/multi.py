"""Requests that need several specialists: each must run and get its own report
section. mx-03 makes the trials server fail: the report must degrade (the
other sections, plus a Note naming what was unavailable), not break."""

from medagent.core.models import Medication, PatientContext
from medagent.core.types import AgentRole, InteractionSeverity
from tests.eval.agent_cases.base import AgentCase

CASES: list[AgentCase] = [
    AgentCase(
        id="mx-01",
        category="multi",
        patient=PatientContext(
            id="P-EVAL-971",
            name="Patient Tau Two",
            age=70,
            sex="F",
            conditions=["heart failure"],
            medications=[Medication(name="Lisinopril"), Medication(name="Spironolactone")],
        ),
        message="Check interactions for Lisinopril and Spironolactone, and what treatment "
        "does the guideline recommend for her heart failure?",
        source="FDA spironolactone label (hyperkalaemia with ACE inhibitors)",
        expected_roles=frozenset({AgentRole.DRUG_SAFETY, AgentRole.EVIDENCE}),
        must_mention=(("spironolactone",), ("potassium", "hyperkal")),
        drug_pair=("Lisinopril", "Spironolactone"),
        min_severity=InteractionSeverity.MODERATE,
        clinical_severity=InteractionSeverity.MAJOR,
    ),
    AgentCase(
        id="mx-02",
        category="multi",
        patient=PatientContext(
            id="P-EVAL-972",
            name="Patient Upsilon Two",
            age=58,
            sex="M",
            conditions=["type 2 diabetes"],
            medications=[Medication(name="Metformin"), Medication(name="Glimepiride")],
        ),
        message="Check interactions for Metformin and Glimepiride, what treatment do the "
        "guidelines recommend, and are there clinical trials he could join?",
        source="Three specialists in one request",
        expected_roles=frozenset(
            {AgentRole.DRUG_SAFETY, AgentRole.EVIDENCE, AgentRole.TRIAL_FINDER}
        ),
        must_mention=(("trial", "study", "nct"),),
        drug_pair=("Metformin", "Glimepiride"),
        min_severity=InteractionSeverity.MODERATE,
    ),
    AgentCase(
        id="mx-03",
        category="multi",
        patient=PatientContext(
            id="P-EVAL-973",
            name="Patient Phi Two",
            age=64,
            sex="F",
            conditions=["type 2 diabetes"],
        ),
        message="What treatment do the guidelines recommend, and are there clinical trials "
        "she could join?",
        source="Degraded report: the trials server is made to fail",
        expected_roles=frozenset({AgentRole.EVIDENCE, AgentRole.TRIAL_FINDER}),
        must_mention=(("metformin", "glucose", "hba1c", "glycemic", "glycaemic"),),
        fail_role=AgentRole.TRIAL_FINDER,
    ),
]
