"""Tracks cumulative LLM token spend across one multi-agent run and records a
step-level trace of what ran, what it cost, and why anything was skipped.

One AgentRunTracker instance covers one call into a workflow (assess or
followup) since specialist agent nodes there run concurrently -- a node can
only observe budget already spent by steps that finished before it started,
not siblings dispatched in the same superstep.
"""

from dataclasses import dataclass, field

from medagent.infra.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentStepRecord:
    name: str
    status: str  # "completed", "skipped" (budget), or "failed"
    tokens_used: int = 0
    detail: str | None = None


@dataclass
class AgentRunTracker:
    max_tokens: int
    run_id: str
    tokens_used: int = 0
    steps: list[AgentStepRecord] = field(default_factory=list)

    def over_budget(self) -> bool:
        return self.tokens_used >= self.max_tokens

    def record(self, step_name: str, usage: dict[str, int] | None = None) -> None:
        tokens = sum((usage or {}).values())
        self.tokens_used += tokens
        self.steps.append(AgentStepRecord(name=step_name, status="completed", tokens_used=tokens))
        logger.info(
            "agent_step_completed",
            run_id=self.run_id,
            step=step_name,
            tokens=tokens,
            cumulative_tokens=self.tokens_used,
            max_tokens=self.max_tokens,
        )

    def skip(self, step_name: str) -> None:
        detail = f"token budget exhausted: {self.tokens_used}/{self.max_tokens}"
        self.steps.append(AgentStepRecord(name=step_name, status="skipped", detail=detail))
        logger.warning(
            "agent_step_skipped",
            run_id=self.run_id,
            step=step_name,
            tokens_used=self.tokens_used,
            max_tokens=self.max_tokens,
        )

    def fail(self, step_name: str, reason: str) -> None:
        self.steps.append(AgentStepRecord(name=step_name, status="failed", detail=reason))
        logger.warning("agent_step_failed", run_id=self.run_id, step=step_name, reason=reason)

    def skipped_steps(self) -> list[str]:
        return [step.name for step in self.steps if step.status == "skipped"]

    def failed_steps(self) -> list[str]:
        return [step.name for step in self.steps if step.status == "failed"]

    def trace_summary(self) -> str:
        return ", ".join(
            f"{step.name}={step.status}({step.tokens_used} tok)" for step in self.steps
        )
