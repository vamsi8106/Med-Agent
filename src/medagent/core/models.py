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


class Visit(BaseModel):
    """A past visit record: what the doctor asked and what was assessed.

    Append-only -- visits are written once (on save) and never updated or
    deleted, so agents reasoning over history see exactly what happened.
    """

    id: str | None = None
    visit_date: datetime
    chief_complaint: str | None = None
    assessment: str | None = None
    plan: str | None = None
    created_at: datetime | None = None


class PatientContext(BaseModel):
    id: str
    name: str
    age: int
    sex: str
    # Owning doctor for per-doctor data isolation. Stamped server-side from
    # the creating doctor's JWT (never client-supplied) and immutable after
    # creation -- see PatientStore.save_patient.
    doctor_id: str | None = None
    weight_kg: float | None = None
    height_cm: float | None = None
    conditions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)
    visits: list[Visit] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: InteractionSeverity
    description: str
    source: str | None = None
    checked_at: datetime


class ToolCall(BaseModel):
    # id ties a tool-result message back to the assistant turn that requested
    # it; providers that do real function calling require it.
    id: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None


class Message(BaseModel):
    role: str
    content: str
    name: str | None = None
    timestamp: datetime | None = None
    # Set on an assistant turn that requested tools, and on the role="tool"
    # messages carrying their results (tool_call_id), so a provider can replay
    # the exact call/result protocol back to the model.
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


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
    usage: dict[str, int] = Field(default_factory=dict)


class AgentFailure(BaseModel):
    """A specialist step that could not complete. `error` is doctor-safe text,
    never a raw exception message (provider errors can embed account ids)."""

    role: AgentRole
    error: str
    # Non-sensitive classification carried alongside the safe text so the caller
    # can tell a rate limit (back off, retry) from an outage.
    reason: str = "error"
    retry_after: float | None = None


class User(BaseModel):
    id: str
    username: str
    role: str = "doctor"
    created_at: datetime | None = None
