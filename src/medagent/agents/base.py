"""Generic ReAct agent: perceive -> think -> act, looped until a final answer.

Implemented as a LangGraph StateGraph: a "think" node calls the LLM, a
conditional edge routes to "act" (execute tool calls, loop back to "think")
or "finalize" (no tool calls left, return the answer) or raises once
max_iterations is exhausted, mirroring the original for-loop's semantics.
"""

from datetime import UTC, datetime
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from medagent.core.exceptions import ToolError
from medagent.core.interfaces import BaseAgent, BaseLLMProvider
from medagent.core.models import LLMResponse, Message, PatientContext, ToolCall
from medagent.infra.logging import get_logger
from medagent.memory.session import SessionMemory
from medagent.tools.registry import ToolRegistry

logger = get_logger(__name__)


class ReActState(TypedDict):
    iterations: int
    tool_calls: list[ToolCall]
    content: str
    final_answer: str


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
        self._graph = self._build_graph().compile()

    async def run(self, context: PatientContext, message: str) -> str:
        self._session.add_message(
            Message(role="user", content=message, timestamp=datetime.now(UTC))
        )
        result = await self._graph.ainvoke(
            {"iterations": 0, "tool_calls": [], "content": "", "final_answer": ""}
        )
        return result["final_answer"]

    def _build_graph(self) -> StateGraph:
        async def think_node(state: ReActState) -> dict[str, object]:
            if state["iterations"] >= self._max_iterations:
                raise ToolError(
                    f"ReAct loop exceeded max_iterations={self._max_iterations} "
                    "without a final answer"
                )
            history = self._perceive()
            response = await self._think(history)
            return {
                "iterations": state["iterations"] + 1,
                "content": response.content,
                "tool_calls": response.tool_calls,
            }

        async def act_node(state: ReActState) -> dict[str, object]:
            await self._act(state["tool_calls"])
            return {}

        async def finalize_node(state: ReActState) -> dict[str, object]:
            self._session.add_message(
                Message(role="assistant", content=state["content"], timestamp=datetime.now(UTC))
            )
            return {"final_answer": state["content"]}

        def route_after_think(state: ReActState) -> str:
            return "finalize_node" if not state["tool_calls"] else "act_node"

        graph = StateGraph(ReActState)
        graph.add_node("think_node", think_node)
        graph.add_node("act_node", act_node)
        graph.add_node("finalize_node", finalize_node)

        graph.add_edge(START, "think_node")
        graph.add_conditional_edges("think_node", route_after_think, ["act_node", "finalize_node"])
        graph.add_edge("act_node", "think_node")
        graph.add_edge("finalize_node", END)
        return graph

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
