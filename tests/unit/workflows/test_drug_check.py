from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from medagent.core.exceptions import PatientNotFoundError
from medagent.core.models import AgentResult, DrugInteraction, Medication, PatientContext
from medagent.core.types import AgentRole, InteractionSeverity
from medagent.workflows.drug_check import run_drug_check


async def test_drug_check_raises_for_unknown_patient() -> None:
    memory = AsyncMock()
    memory.get_patient.return_value = None

    with pytest.raises(PatientNotFoundError):
        await run_drug_check(AsyncMock(), memory, "P-UNKNOWN", "Ibuprofen")


async def test_drug_check_builds_message_from_current_meds() -> None:
    memory = AsyncMock()
    memory.get_patient.return_value = PatientContext(
        id="P-TEST-700",
        name="Patient Alpha",
        age=50,
        sex="F",
        medications=[Medication(name="Metformin")],
    )
    interaction = DrugInteraction(
        drug_a="Metformin",
        drug_b="Glimepiride",
        severity=InteractionSeverity.MODERATE,
        description="Additive glucose lowering.",
        source="med-research-mcp-suite",
        checked_at=datetime.now(UTC),
    )
    drug_safety = AsyncMock()
    drug_safety.run_result.return_value = AgentResult(
        role=AgentRole.DRUG_SAFETY, summary="No major concerns.", interactions=[interaction]
    )

    outcome = await run_drug_check(drug_safety, memory, "P-TEST-700", "Glimepiride")

    drug_safety.run_result.assert_awaited_once()
    args, _ = drug_safety.run_result.call_args
    assert "Metformin" in args[1]
    assert "Glimepiride" in args[1]
    assert outcome.answer == "No major concerns."
    assert outcome.interactions == [interaction]


async def test_drug_check_forwards_doctor_id_to_memory() -> None:
    memory = AsyncMock()
    memory.get_patient.return_value = PatientContext(
        id="P-TEST-701", name="Patient Beta", age=40, sex="M", doctor_id="DR-TEST-001"
    )
    drug_safety = AsyncMock()
    drug_safety.run_result.return_value = AgentResult(
        role=AgentRole.DRUG_SAFETY, summary="No major concerns."
    )

    await run_drug_check(drug_safety, memory, "P-TEST-701", "Ibuprofen", doctor_id="DR-TEST-001")

    memory.get_patient.assert_awaited_once_with("P-TEST-701", "DR-TEST-001")
