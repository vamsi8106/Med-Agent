"""Phase 4 gate: a follow-up visit recalls the patient and retrieves matching guidelines.

Uses the real local sentence-transformers model, a real (throwaway) Postgres
container (via the shared `pg_dsn` fixture in tests/conftest.py -- skipped if
Docker isn't available), and a real in-memory ChromaDB instance -- not
mocked -- since this exercises the actual RAG pipeline end to end. Set
HF_HUB_OFFLINE so it never reaches out to the network if the model is already
cached locally.
"""

import os
from unittest.mock import patch

import chromadb

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from medagent.core.models import Medication, PatientContext  # noqa: E402
from medagent.memory.patient_store import PatientStore  # noqa: E402
from medagent.memory.persistent import PersistentStore  # noqa: E402
from medagent.rag.embeddings import EmbeddingModel  # noqa: E402
from medagent.rag.pipeline import IngestionPipeline  # noqa: E402
from medagent.rag.retriever import GuidelineRetrieverTool  # noqa: E402
from medagent.rag.vector_store import VectorStore  # noqa: E402


async def test_followup_visit_recalls_patient_and_retrieves_guidelines(pg_dsn: str) -> None:
    persistent = PersistentStore(pg_dsn)
    await persistent.init_schema()
    patient_store = PatientStore(persistent)

    context = PatientContext(
        id="P-TEST-100",
        name="Patient Alpha",
        age=62,
        sex="F",
        conditions=["type 2 diabetes"],
        medications=[Medication(name="Metformin", dose="500mg")],
    )
    await patient_store.save_patient(context)

    embeddings = EmbeddingModel()
    # VectorStore always constructs chromadb.HttpClient (production talks to the
    # chromadb server container) -- redirect to an in-memory client here since
    # this integration test has no running Chroma server.
    with patch(
        "medagent.rag.vector_store.chromadb.HttpClient", return_value=chromadb.EphemeralClient()
    ):
        vector_store = VectorStore(host="unused", port=0)
    pipeline = IngestionPipeline(embeddings=embeddings, vector_store=vector_store)
    await pipeline.ingest_text(
        "# Pharmacotherapy\nMetformin is recommended as first-line therapy for "
        "type 2 diabetes mellitus unless contraindicated.",
        source_name="ada-2024-guideline",
    )

    recalled = await patient_store.get_patient("P-TEST-100")
    assert recalled is not None
    assert recalled.name == "Patient Alpha"
    assert recalled.medications[0].name == "Metformin"

    retriever = GuidelineRetrieverTool(embeddings=embeddings, vector_store=vector_store)
    evidence = await retriever.run(query="first-line therapy for type 2 diabetes", top_k=1)

    assert len(evidence) == 1
    assert "metformin" in evidence[0].summary.lower()
    assert evidence[0].source == "ada-2024-guideline"
