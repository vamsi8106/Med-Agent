from datetime import UTC, datetime

from medagent.core.models import LabResult
from medagent.tools.custom.lab_interpreter import LabInterpreterTool


def _lab(value: float, low: float | None, high: float | None) -> LabResult:
    return LabResult(
        test_name="Glucose",
        value=value,
        unit="mg/dL",
        reference_low=low,
        reference_high=high,
        collected_at=datetime.now(UTC),
    )


async def test_flags_high_value() -> None:
    tool = LabInterpreterTool()
    result = await tool.run([_lab(200, 70, 140)])
    assert result.data[0].flag == "high"


async def test_flags_low_value() -> None:
    tool = LabInterpreterTool()
    result = await tool.run([_lab(50, 70, 140)])
    assert result.data[0].flag == "low"


async def test_flags_normal_value() -> None:
    tool = LabInterpreterTool()
    result = await tool.run([_lab(100, 70, 140)])
    assert result.data[0].flag == "normal"
