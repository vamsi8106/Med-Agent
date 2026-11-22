"""Trial search: routed to the Trial Finder, which searches ClinicalTrials.gov
for the first recorded condition and summarises what it finds."""

from medagent.core.models import PatientContext
from medagent.core.types import AgentRole
from tests.eval.agent_cases.base import AgentCase


def _trial_case(case_id: str, name: str, age: int, sex: str, condition: str) -> AgentCase:
    return AgentCase(
        id=case_id,
        category="trials",
        patient=PatientContext(
            id=case_id.replace("tr-", "P-EVAL-7"),
            name=name,
            age=age,
            sex=sex,
            conditions=[condition],
        ),
        message="Are there any clinical trials this patient could join?",
        source="ClinicalTrials.gov via healthcare-mcp",
        expected_roles=frozenset({AgentRole.TRIAL_FINDER}),
        must_mention=(("trial", "study", "studies", "nct"),),
    )


CASES: list[AgentCase] = [
    _trial_case("tr-01", "Patient Beta Two", 60, "F", "type 2 diabetes"),
    _trial_case("tr-02", "Patient Gamma Two", 67, "M", "chronic kidney disease"),
    _trial_case("tr-03", "Patient Delta Two", 54, "F", "hypertension"),
    _trial_case("tr-04", "Patient Epsilon Two", 72, "M", "heart failure"),
]
