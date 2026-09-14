from unittest.mock import AsyncMock

import pytest

from medagent.core.exceptions import PatientNotFoundError
from medagent.core.models import Medication, PatientContext
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
    drug_safety = AsyncMock()
    drug_safety.run.return_value = "No major concerns."

    result = await run_drug_check(drug_safety, memory, "P-TEST-700", "Glimepiride")

    drug_safety.run.assert_awaited_once()
    args, _ = drug_safety.run.call_args
    assert "Metformin" in args[1]
    assert "Glimepiride" in args[1]
    assert result == "No major concerns."
