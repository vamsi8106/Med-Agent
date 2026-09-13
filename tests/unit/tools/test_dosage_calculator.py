from medagent.core.models import Medication, PatientContext
from medagent.core.types import CKDStage
from medagent.tools.custom.dosage_calculator import DosageCalculatorTool


def _patient() -> PatientContext:
    return PatientContext(id="P-TEST-001", name="Patient Alpha", age=70, sex="F")


async def test_no_ckd_stage_means_no_adjustment() -> None:
    tool = DosageCalculatorTool()
    result = await tool.run(Medication(name="Metformin"), _patient())
    assert result.data["reduction_pct"] == 0.0


async def test_stage_5_ckd_applies_largest_reduction() -> None:
    tool = DosageCalculatorTool()
    result = await tool.run(Medication(name="Metformin"), _patient(), ckd_stage=CKDStage.STAGE_5)
    assert result.data["reduction_pct"] == 0.75
    assert result.data["ckd_stage"] == "stage_5"
