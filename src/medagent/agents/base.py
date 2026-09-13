"""Generic ReAct agent: perceive -> think -> act, looped until a final answer."""

from datetime import UTC, datetime

from medagent.core.exceptions import ToolError
from medagent.core.interfaces import BaseAgent, BaseLLMProvider
from medagent.core.models import LLMResponse, Message, PatientContext, ToolCall
from medagent.infra.logging import get_logger
from medagent.memory.session import SessionMemory
from medagent.tools.registry import ToolRegistry

logger = get_logger(__name__)


class ReActAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLMProvider,
        tools: ToolRegistry,
        session: SessionMemory,
        system_prompt: str,
        max_iterations: int = 5,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._session = session
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations

    async def run(self, context: PatientContext, message: str) -> str:
        self._session.add_message(
            Message(role="user", content=message, timestamp=datetime.now(UTC))
        )

        for _ in range(self._max_iterations):
            history = self._perceive()
            response = await self._think(history)

            if not response.tool_calls:
                self._session.add_message(
                    Message(role="assistant", content=response.content, timestamp=datetime.now(UTC))
                )
                return response.content

            await self._act(response.tool_calls)

        raise ToolError(
            f"ReAct loop exceeded max_iterations={self._max_iterations} without a final answer"
        )

    def _perceive(self) -> list[Message]:
        system = Message(role="system", content=self._system_prompt)
        return [system, *self._session.get_messages()]

    async def _think(self, history: list[Message]) -> LLMResponse:
        return await self._llm.complete(history, tools=self._tools.schemas())

    async def _act(self, tool_calls: list[ToolCall]) -> None:
        for call in tool_calls:
            tool = self._tools.get(call.tool_name)
            result = await tool.run(**call.arguments)
            call.result = result
            self._session.add_message(
                Message(
                    role="tool",
                    name=call.tool_name,
                    content=str(result),
                    timestamp=datetime.now(UTC),
                )
            )
