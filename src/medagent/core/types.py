"""Enums shared across MedAgent modules."""

from enum import StrEnum


class EvidenceGrade(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class InteractionSeverity(StrEnum):
    CONTRAINDICATED = "contraindicated"
    MAJOR = "major"
    MODERATE = "moderate"
    MINOR = "minor"
    NONE = "none"


class AgentRole(StrEnum):
    TRIAGE = "triage"
    DRUG_SAFETY = "drug_safety"
    EVIDENCE = "evidence"
    TRIAL_FINDER = "trial_finder"
    REPORT = "report"
