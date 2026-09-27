"""Synthesizes multiple agents' findings into a structured markdown report."""

from medagent.core.models import AgentResult, LabFlag, PatientContext, RecordChanges
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


def format_changes(changes: RecordChanges) -> list[str]:
    """One line per change, values only. Shared with the evidence agent's prompt."""
    lines = [f"- Started: {name}" for name in changes.medications_started]
    lines += [f"- Stopped: {name}" for name in changes.medications_stopped]
    lines += [f"- Changed: {c.name}: {c.before} -> {c.after}" for c in changes.medications_changed]
    lines += [f"- New condition: {c}" for c in changes.conditions_added]
    lines += [f"- Condition removed: {c}" for c in changes.conditions_removed]
    lines += [f"- New allergy: {a}" for a in changes.allergies_added]
    lines += [f"- Allergy removed: {a}" for a in changes.allergies_removed]
    for lab in changes.lab_changes:
        before = f"{lab.previous_value} -> " if lab.previous_value is not None else "new: "
        reference = (
            f" (reference {lab.reference_low}-{lab.reference_high})"
            if lab.reference_low is not None and lab.reference_high is not None
            else ""
        )
        lines.append(
            f"- {lab.test_name}: {before}{lab.current_value} {lab.unit}{reference}, "
            f"collected {lab.collected_at.date()}"
        )
    return lines


def _format_changes_section(changes: RecordChanges | None) -> str | None:
    if changes is None:
        return None
    header = f"## Changes Since Last Visit ({changes.since.date()})"
    if changes.is_empty:
        return f"{header}\n\nNo recorded changes to medications, conditions, allergies or labs."
    return "\n".join([header, "", *format_changes(changes)])


class ReportGeneratorTool(BaseTool):
    name = "report_generator"
    description = "Synthesizes multi-agent findings into a structured markdown report."

    def __init__(self, lab_interpreter: LabInterpreterTool | None = None) -> None:
        self._lab_interpreter = lab_interpreter or LabInterpreterTool()

    # Narrowed from BaseTool.run(**kwargs: Any) to this tool's real, typed
    # params -- an intentional, safe narrowing, not a real LSP mismatch.
    async def run(  # type: ignore[override]
        self, patient: PatientContext, results: list[AgentResult]
    ) -> ToolResult:
        header = f"# Clinical Report: {patient.name} ({patient.id})"
        sections = [_format_section(result) for result in results]

        # Deterministic, always-shown facts from the patient's own record --
        # not gated on whether an LLM-driven agent happened to mention them.
        changes_section = _format_changes_section(patient.changes_since_last_visit)
        if changes_section:
            sections.append(changes_section)

        allergies_section = _format_allergies_section(patient)
        if allergies_section:
            sections.append(allergies_section)

        if patient.lab_results:
            lab_flag_result = await self._lab_interpreter.run(patient.lab_results, patient)
            # ToolResult.data is intentionally untyped (Any); LabInterpreterTool
            # always puts a list[LabFlag] there.
            assert isinstance(lab_flag_result.data, list)
            lab_flags_section = _format_lab_flags_section(lab_flag_result.data)
            if lab_flags_section:
                sections.append(lab_flags_section)

        report = "\n\n".join([header, *sections])
        return ToolResult(tool_name=self.name, success=True, data=report)
