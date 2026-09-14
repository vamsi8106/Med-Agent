"""Synthesizes multiple agents' findings into a structured markdown report."""

from medagent.core.models import AgentResult, LabFlag, PatientContext
from medagent.tools.base import BaseTool, ToolResult
from medagent.tools.custom.lab_interpreter import LabInterpreterTool


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


def _format_allergies_section(patient: PatientContext) -> str | None:
    if not patient.allergies:
        return None
    lines = ["## Known Allergies", ""]
    lines.extend(f"- {allergy}" for allergy in patient.allergies)
    return "\n".join(lines)


def _format_lab_flags_section(flags: list[LabFlag]) -> str | None:
    abnormal = [flag for flag in flags if flag.flag != "normal"]
    if not abnormal:
        return None
    lines = ["## Lab Flags", ""]
    lines.extend(
        f"- {flag.test_name}: {flag.value} {flag.unit} ({flag.flag.upper()}"
        + (f", reference {flag.reference_range}" if flag.reference_range else "")
        + ")"
        for flag in abnormal
    )
    return "\n".join(lines)


class ReportGeneratorTool(BaseTool):
    name = "report_generator"
    description = "Synthesizes multi-agent findings into a structured markdown report."

    def __init__(self, lab_interpreter: LabInterpreterTool | None = None) -> None:
        self._lab_interpreter = lab_interpreter or LabInterpreterTool()

    async def run(self, patient: PatientContext, results: list[AgentResult]) -> ToolResult:
        header = f"# Clinical Report: {patient.name} ({patient.id})"
        sections = [_format_section(result) for result in results]

        # Deterministic, always-shown facts from the patient's own record --
        # not gated on whether an LLM-driven agent happened to mention them.
        allergies_section = _format_allergies_section(patient)
        if allergies_section:
            sections.append(allergies_section)

        if patient.lab_results:
            lab_flag_result = await self._lab_interpreter.run(patient.lab_results, patient)
            lab_flags_section = _format_lab_flags_section(lab_flag_result.data)
            if lab_flags_section:
                sections.append(lab_flags_section)

        report = "\n\n".join([header, *sections])
        return ToolResult(tool_name=self.name, success=True, data=report)
