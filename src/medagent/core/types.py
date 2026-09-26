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


class CKDStage(StrEnum):
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    STAGE_3A = "stage_3a"
    STAGE_3B = "stage_3b"
    STAGE_4 = "stage_4"
    STAGE_5 = "stage_5"


class TrialPhase(StrEnum):
    EARLY_PHASE_1 = "early_phase_1"
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    PHASE_4 = "phase_4"
    NOT_APPLICABLE = "not_applicable"


class AgentRole(StrEnum):
    TRIAGE = "triage"
    DRUG_SAFETY = "drug_safety"
    EVIDENCE = "evidence"
    TRIAL_FINDER = "trial_finder"
    REPORT = "report"
