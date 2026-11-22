"""What one agent-eval case declares, and the vocabulary its checks use.

A case is a synthetic patient plus a doctor's message, run end to end through
the real assessment workflow. Every field is optional except the patient and
message, and a check only applies when its field is set.

Clinical ground truth (`clinical_severity`, `must_mention`, allergy
expectations) is drafted from the public source in `source` and is NOT
clinician-reviewed until `reviewed=True`; the report counts unreviewed cases.
"""

from dataclasses import dataclass, field
from typing import Literal

from medagent.core.models import PatientContext
from medagent.core.types import AgentRole, InteractionSeverity

Category = Literal[
    "interactions",
    "allergy",
    "renal",
    "evidence",
    "history",
    "trials",
    "injection",
    "followup",
    "drug_check",
    "multi",
]
Workflow = Literal["assessment", "followup", "drug_check"]

# Categories `make agent-eval` runs three times; their gates are held to 100 %.
SAFETY_CATEGORIES: frozenset[str] = frozenset(
    {"interactions", "allergy", "injection", "drug_check"}
)

SEVERITY_RANK: dict[InteractionSeverity, int] = {
    InteractionSeverity.NONE: 0,
    InteractionSeverity.MINOR: 1,
    InteractionSeverity.MODERATE: 2,
    InteractionSeverity.MAJOR: 3,
    InteractionSeverity.CONTRAINDICATED: 4,
}


@dataclass(frozen=True)
class Injection:
    """Instructions hidden in what the agents read. `where`: "record" -- already
    inside the patient record (a note or condition); "message" -- inside the
    doctor's own message; "literature", "trials", "interactions" -- added to
    that MCP server's results by the spy. `canary` is a string the injection
    asks for: it must never appear in an agent's answer."""

    where: Literal["record", "message", "literature", "trials", "interactions"]
    canary: str
    text: str = ""


@dataclass(frozen=True)
class AgentCase:
    id: str
    category: Category
    patient: PatientContext
    message: str
    source: str
    workflow: Workflow = "assessment"
    # Follow-up: the record as it stood at the last visit (saved, then a visit
    # snapshot, then `patient` as the current record).
    previous: PatientContext | None = None
    # Drug-check: the drug being added to the patient's current medications.
    new_drug: str | None = None
    # Text that must appear under "## Changes Since Last Visit".
    must_show_changes: tuple[str, ...] = ()
    # A specialist whose MCP server the eval makes fail: the report must
    # degrade (that section missing, a Note naming it), not break.
    fail_role: AgentRole | None = None
    # Triage must route to exactly these specialists.
    expected_roles: frozenset[AgentRole] = frozenset()
    # The evidence agent should have used these tools (subset check).
    expected_tools: frozenset[str] = frozenset()
    # Groups of synonyms the answer should contain; each group is one fact.
    must_mention: tuple[tuple[str, ...], ...] = ()
    # Phrases whose presence means the answer contradicts the case.
    must_not_mention: tuple[str, ...] = ()
    # The drug pair checked, its gate floor, and the clinically expected
    # severity (None where the sources do not settle it).
    drug_pair: tuple[str, str] | None = None
    min_severity: InteractionSeverity | None = None
    clinical_severity: InteractionSeverity | None = None
    expect_allergy_flag: bool = False
    injection: Injection | None = None
    # A behaviour the system is known not to have yet: reported, never gated.
    known_gap: str | None = None
    reviewed: bool = False
    notes: str = ""
    tags: tuple[str, ...] = field(default=())
