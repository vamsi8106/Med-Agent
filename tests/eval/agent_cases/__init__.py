"""All agent-eval cases. A new agent or behaviour gets cases here (see base.py
for the fields and the checks they switch on)."""

from tests.eval.agent_cases import (
    allergy,
    drug_check,
    evidence,
    followup,
    history,
    injection,
    interactions,
    multi,
    renal,
    trials,
)
from tests.eval.agent_cases.base import SAFETY_CATEGORIES, AgentCase, Category, Injection

ALL_CASES: list[AgentCase] = [
    *interactions.CASES,
    *allergy.CASES,
    *renal.CASES,
    *evidence.CASES,
    *history.CASES,
    *trials.CASES,
    *injection.CASES,
    *followup.CASES,
    *drug_check.CASES,
    *multi.CASES,
]

__all__ = ["ALL_CASES", "SAFETY_CATEGORIES", "AgentCase", "Category", "Injection"]
