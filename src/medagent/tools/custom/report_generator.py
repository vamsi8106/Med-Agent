"""Synthesizes multiple agents' findings into a structured markdown report."""

from medagent.core.models import AgentResult, PatientContext
from medagent.tools.base import BaseTool, ToolResult


def _format_section(result: AgentResult) -> str:
    title = result.role.value.replace("_", " ").title()
    lines = [f"## {title}", "", result.summary]

    if result.interactions:
        lines.append("")
        lines.append("**Drug Interactions:**")
        lines.extend(
            f"- {i.drug_a} + {i.drug_b}: {i.severity.value} (source: {i.source or 'unknown'})"
            for i in result.interactions
        )

    if result.evidence:
        lines.append("")
        lines.append("**Evidence:**")
        lines.extend(f"- {e.title} (source: {e.source or 'unknown'})" for e in result.evidence)

    return "\n".join(lines)


class ReportGeneratorTool(BaseTool):
    name = "report_generator"
    description = "Synthesizes multi-agent findings into a structured markdown report."

    async def run(self, patient: PatientContext, results: list[AgentResult]) -> ToolResult:
        header = f"# Clinical Report: {patient.name} ({patient.id})"
        sections = [_format_section(result) for result in results]
        report = "\n\n".join([header, *sections])
        return ToolResult(tool_name=self.name, success=True, data=report)
