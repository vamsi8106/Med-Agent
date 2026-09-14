"""Drug-check workflow: load current meds from memory, check a proposed addition."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.core.exceptions import PatientNotFoundError
from medagent.core.interfaces import BaseMemory
from medagent.core.models import PatientContext


class DrugCheckState(TypedDict):
    patient_id: str
    new_drug: str
    context: PatientContext | None
    message: str
    answer: str


def _build_graph(drug_safety: DrugSafetyAgent, memory: BaseMemory) -> StateGraph:
    async def load_patient(state: DrugCheckState) -> dict[str, object]:
        context = await memory.get_patient(state["patient_id"])
        if context is None:
            raise PatientNotFoundError(f"No existing patient record for id: {state['patient_id']}")
        return {"context": context}

    async def check_interactions(state: DrugCheckState) -> dict[str, object]:
        context = state["context"]
        assert context is not None
        current_drugs = [med.name for med in context.medications]
        message = f"Check interactions for {', '.join([*current_drugs, state['new_drug']])}"
        answer = await drug_safety.run(context, message)
        return {"message": message, "answer": answer}

    graph = StateGraph(DrugCheckState)
    graph.add_node("load_patient", load_patient)
    graph.add_node("check_interactions", check_interactions)
    graph.add_edge(START, "load_patient")
    graph.add_edge("load_patient", "check_interactions")
    graph.add_edge("check_interactions", END)
    return graph


async def run_drug_check(
    drug_safety: DrugSafetyAgent, memory: BaseMemory, patient_id: str, new_drug: str
) -> str:
    graph = _build_graph(drug_safety, memory).compile()
    result = await graph.ainvoke({"patient_id": patient_id, "new_drug": new_drug})
    return result["answer"]
