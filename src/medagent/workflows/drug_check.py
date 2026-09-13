"""Drug-check workflow: load current meds from memory, check a proposed addition."""

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.core.exceptions import MedAgentError
from medagent.core.interfaces import BaseMemory


async def run_drug_check(
    drug_safety: DrugSafetyAgent, memory: BaseMemory, patient_id: str, new_drug: str
) -> str:
    context = await memory.get_patient(patient_id)
    if context is None:
        raise MedAgentError(f"No existing patient record for id: {patient_id}")

    current_drugs = [med.name for med in context.medications]
    message = f"Check interactions for {', '.join([*current_drugs, new_drug])}"
    return await drug_safety.run(context, message)
