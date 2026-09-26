"""Generic ReAct agent: perceive -> think -> act, looped until a final answer.

Implemented as a LangGraph StateGraph: a "think" node calls the LLM, a
conditional edge routes to "act" (execute tool calls, loop back to "think")
or "finalize" (no tool calls left, return the answer). The loop is bounded
three ways so a misbehaving model can't run away: max_iterations, an optional
prompt-size ceiling checked before every LLM call, and a per-result cap on
tool output. Tool failures are returned to the model as ordinary tool results
so it can correct itself, rather than aborting the run.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from medagent.core.config import get_settings
from medagent.core.exceptions import ToolError
from medagent.core.interfaces import BaseAgent, BaseLLMProvider
from medagent.core.models import LLMResponse, Message, PatientContext, ToolCall
from medagent.infra.context_budget import approx_token_count, truncate_text
from medagent.infra.guardrails import wrap_untrusted
from medagent.infra.logging import get_logger
from medagent.memory.session import SessionMemory
from medagent.tools.base import ToolResult
from medagent.tools.registry import ToolRegistry

logger = get_logger(__name__)

_FINAL_ANSWER_DIRECTIVE = (
    "You have gathered enough evidence. Write your final answer now, using only the tool "
    "results above and citing your sources. Do not call any more tools."
)


class ReActState(TypedDict):
    iterations: int
    tool_calls: list[ToolCall]
    executed: list[ToolCall]
    content: str
    final_answer: str
    usage: dict[str, int]


@dataclass
class ReActResult:
    answer: str
    tool_calls: list[ToolCall]
    usage: dict[str, int]
    iterations: int


def _merge_usage(total: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    merged = dict(total)
    for key, value in new.items():
        merged[key] = merged.get(key, 0) + value
    return merged


class ReActAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLMProvider,
        tools: ToolRegistry,
        session: SessionMemory,
        system_prompt: str,
        max_iterations: int = 5,
        max_prompt_tokens: int | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._session = session
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._max_prompt_tokens = max_prompt_tokens
        self._graph = self._build_graph().compile()

    async def run(self, context: PatientContext, message: str) -> str:
        return (await self.run_detailed(context, message)).answer

    async def run_detailed(self, context: PatientContext, message: str) -> ReActResult:
        self._session.add_message(
            Message(role="user", content=message, timestamp=datetime.now(UTC))
        )
        result = await self._graph.ainvoke(
            {
                "iterations": 0,
                "tool_calls": [],
                "executed": [],
                "content": "",
                "final_answer": "",
                "usage": {},
            }
        )
        return ReActResult(
            answer=result["final_answer"],
            tool_calls=result["executed"],
            usage=result["usage"],
            iterations=result["iterations"],
        )

    def _build_graph(self) -> StateGraph:
        async def think_node(state: ReActState) -> dict[str, object]:
            if state["iterations"] >= self._max_iterations:
                raise ToolError(
                    f"ReAct loop exceeded max_iterations={self._max_iterations} "
                    "without a final answer"
                )
            history = self._perceive()
            self._enforce_prompt_budget(history)
            # On the last allowed turn, tell the model to write its answer from
            # what it already gathered -- otherwise a model that keeps
            # searching burns every iteration and the run aborts with nothing
            # to show for the tokens spent. The tools stay attached: omitting
            # them makes Groq reject the request outright if the model tries
            # one more call anyway. The directive is transient, never stored
            # in the session history.
            is_last_turn = state["iterations"] == self._max_iterations - 1
            if is_last_turn and state["executed"]:
                history = [*history, Message(role="user", content=_FINAL_ANSWER_DIRECTIVE)]
            response = await self._think(history)
            if response.tool_calls:
                # The assistant turn that requested the tools must be in the
                # history before their results, or a real provider rejects
                # the tool messages as orphaned.
                self._session.add_message(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                        timestamp=datetime.now(UTC),
                    )
                )
            return {
                "iterations": state["iterations"] + 1,
                "content": response.content,
                "tool_calls": response.tool_calls,
                "usage": _merge_usage(state["usage"], response.usage),
            }

        async def act_node(state: ReActState) -> dict[str, object]:
            await self._act(state["tool_calls"])
            return {"executed": [*state["executed"], *state["tool_calls"]]}

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

    def _enforce_prompt_budget(self, history: list[Message]) -> None:
        if self._max_prompt_tokens is None:
            return
        used = approx_token_count("\n".join(m.content for m in history))
        if used > self._max_prompt_tokens:
            raise ToolError(
                f"ReAct prompt reached {used} tokens, over the {self._max_prompt_tokens} budget"
            )

    async def _think(self, history: list[Message]) -> LLMResponse:
        return await self._llm.complete(history, tools=self._tools.schemas())

    async def _act(self, tool_calls: list[ToolCall]) -> None:
        for call in tool_calls:
            content = await self._execute(call)
            self._session.add_message(
                Message(
                    role="tool",
                    name=call.tool_name,
                    tool_call_id=call.id,
                    content=content,
                    timestamp=datetime.now(UTC),
                )
            )

    async def _execute(self, call: ToolCall) -> str:
        """Runs one tool call and returns its result as text for the model.

        Every failure mode -- unknown tool, bad arguments, a tool reporting
        failure -- comes back as an "Error: ..." result the model can read and
        retry from. Successful output is capped and delimited as untrusted
        data, since it originates outside this system.
        """
        try:
            tool = self._tools.get(call.tool_name)
            result = await tool.run(**call.arguments)
        except ToolError as exc:
            text = f"Error: {exc}"
        except TypeError as exc:
            text = f"Error: invalid arguments for {call.tool_name}: {exc}"
        else:
            call.result = result
            if isinstance(result, ToolResult):
                text = str(result.data) if result.success else f"Error: {result.error}"
            else:
                text = str(result)

        logger.info("react_tool_executed", tool=call.tool_name, is_error=text.startswith("Error:"))
        if text.startswith("Error:"):
            return text
        capped = truncate_text(
            text, get_settings().agent_context_field_max_tokens, source=f"react.{call.tool_name}"
        )
        return wrap_untrusted(capped)
