"""Core pydantic data models shared across MedAgent modules."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from medagent.core.types import AgentRole, EvidenceGrade, InteractionSeverity


class Medication(BaseModel):
    id: str | None = None
    name: str
    brand_name: str | None = None
    dose: str | None = None
    frequency: str | None = None
    route: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    status: str = "active"


class LabResult(BaseModel):
    id: str | None = None
    test_name: str
    value: float
    unit: str
    reference_low: float | None = None
    reference_high: float | None = None
    is_abnormal: bool = False
    collected_at: datetime


class PatientContext(BaseModel):
    id: str
    name: str
    age: int
    sex: str
    weight_kg: float | None = None
    height_cm: float | None = None
    conditions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: InteractionSeverity
    description: str
    source: str | None = None
    checked_at: datetime


class Message(BaseModel):
    role: str
    content: str
    name: str | None = None
    timestamp: datetime | None = None


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None


class LLMResponse(BaseModel):
    content: str
    model: str
    role_hint: AgentRole | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)


class ClinicalEvidence(BaseModel):
    title: str
    summary: str
    grade: EvidenceGrade | None = None
    source: str | None = None
    url: str | None = None


class LabFlag(BaseModel):
    test_name: str
    value: float
    unit: str
    flag: str
    reference_range: str | None = None
    explanation: str | None = None


class AgentResult(BaseModel):
    role: AgentRole
    summary: str
    evidence: list[ClinicalEvidence] = Field(default_factory=list)
    interactions: list[DrugInteraction] = Field(default_factory=list)


class User(BaseModel):
    id: str
    username: str
    role: str = "doctor"
    created_at: datetime | None = None
