"""Abstract base classes defining the seams between layers."""

from abc import ABC, abstractmethod
from typing import Any

from medagent.core.models import LLMResponse, Message, PatientContext


class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse: ...


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any: ...


class BaseAgent(ABC):
    @abstractmethod
    async def run(self, context: PatientContext, message: str) -> str: ...


class BaseKeyedLimiter(ABC):
    """An abuse limit per key (a username, a client IP, a user id)."""

    @abstractmethod
    async def try_acquire(self, key: str) -> float | None:
        """None when allowed, otherwise the seconds until a token is available.
        Never waits."""


class BaseMemory(ABC):
    @abstractmethod
    async def get_patient(
        self, patient_id: str, doctor_id: str | None = None
    ) -> PatientContext | None: ...

    @abstractmethod
    async def save_patient(self, context: PatientContext) -> None: ...
