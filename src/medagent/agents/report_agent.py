"""Report Agent: synthesizes multiple specialist AgentResults into one report.

Not a BaseAgent subclass: its contract is (patient, list[AgentResult]) -> str,
aggregating several agents' output for a single visit, which doesn't fit
BaseAgent's single (context, message) -> str turn contract.
"""

from medagent.core.exceptions import ToolError
from medagent.core.models import AgentResult, PatientContext
from medagent.tools.custom.lab_interpreter import LabInterpreterTool
from medagent.tools.custom.report_generator import ReportGeneratorTool


class ReportAgent:
    def __init__(
        self,
        report_generator: ReportGeneratorTool | None = None,
        lab_interpreter: LabInterpreterTool | None = None,
    ) -> None:
        self._report_generator = report_generator or ReportGeneratorTool(lab_interpreter)

    async def generate(self, patient: PatientContext, results: list[AgentResult]) -> str:
        result = await self._report_generator.run(patient, results)
        if not result.success:
            raise ToolError(f"Report generation failed: {result.error}")
        return result.data
