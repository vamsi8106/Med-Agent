"""Renal/hepatic dose-adjustment estimator.

Illustrative only: applies a generic percentage reduction by CKD stage, not a
drug-specific pharmacokinetic table. Real dosing must be verified against the
drug's own renal/hepatic dosing guidance (e.g. FDA label, Lexicomp) before use.
"""

from medagent.core.models import Medication, PatientContext
from medagent.core.types import CKDStage
from medagent.tools.base import BaseTool, ToolResult

_CKD_DOSE_REDUCTION_PCT: dict[CKDStage, float] = {
    CKDStage.STAGE_1: 0.0,
    CKDStage.STAGE_2: 0.0,
    CKDStage.STAGE_3A: 0.25,
    CKDStage.STAGE_3B: 0.5,
    CKDStage.STAGE_4: 0.5,
    CKDStage.STAGE_5: 0.75,
}


class DosageCalculatorTool(BaseTool):
    name = "dosage_calculator"
    description = (
        "Estimates a renal-function-adjusted dose reduction percentage. "
        "Illustrative only, not a substitute for drug-specific dosing guidance."
    )

    async def run(
        self, medication: Medication, context: PatientContext, ckd_stage: CKDStage | None = None
    ) -> ToolResult:
        if ckd_stage is None:
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "medication": medication.name,
                    "reduction_pct": 0.0,
                    "note": "No CKD stage provided; no renal adjustment applied.",
                },
            )

        reduction_pct = _CKD_DOSE_REDUCTION_PCT[ckd_stage]
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "medication": medication.name,
                "ckd_stage": ckd_stage.value,
                "reduction_pct": reduction_pct,
                "note": ("Generic estimate; verify against drug-specific renal dosing guidance."),
            },
        )
