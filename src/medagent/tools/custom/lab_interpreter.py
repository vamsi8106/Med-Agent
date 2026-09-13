"""Flags abnormal lab results relative to reference ranges."""

from medagent.core.models import LabFlag, LabResult, PatientContext
from medagent.tools.base import BaseTool, ToolResult


def _flag_for(lab: LabResult) -> str:
    if lab.reference_low is not None and lab.value < lab.reference_low:
        return "low"
    if lab.reference_high is not None and lab.value > lab.reference_high:
        return "high"
    return "normal"


def _reference_range(lab: LabResult) -> str | None:
    if lab.reference_low is None and lab.reference_high is None:
        return None
    return f"{lab.reference_low}-{lab.reference_high} {lab.unit}"


class LabInterpreterTool(BaseTool):
    name = "lab_interpreter"
    description = "Flags lab results as high/low/normal against their reference ranges."

    async def run(
        self, lab_results: list[LabResult], context: PatientContext | None = None
    ) -> ToolResult:
        flags = [
            LabFlag(
                test_name=lab.test_name,
                value=lab.value,
                unit=lab.unit,
                flag=_flag_for(lab),
                reference_range=_reference_range(lab),
            )
            for lab in lab_results
        ]
        return ToolResult(tool_name=self.name, success=True, data=flags)
