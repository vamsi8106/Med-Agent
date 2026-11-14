"""Retrieval eval: how often does guideline search put the right passage on top?

Builds an in-memory ChromaDB index from the local guideline PDFs once per
chunking configuration, asks every question in retrieval_golden_dataset.py, and
reports recall@k and MRR. Run `make rag-eval` before and after any change to
chunking, the embedding model or retrieval, and compare.

- recall@k: share of questions whose answer passage is in the top k results.
- MRR (mean reciprocal rank): average of 1/rank of the first correct result
  (1.0 = always first; 0 when it is not in the top 5).
- reaches model: share of questions whose answer passage survives into what
  the evidence agent actually hands the LLM -- the top 3 passages, joined and
  capped at AGENT_CONTEXT_FIELD_MAX_TOKENS exactly as `search_guidelines` does.
  Recall@k alone flatters big chunks: 5 chunks of 512 words are ~2,300 words,
  most of which the agent cuts before the model sees it.
- beyond limit: share of embedded tokens past the model's limit, which the
  model silently ignores.

Needs the PDFs in guidelines_pdf/ (gitignored) and the local embedding model.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from pypdf import PdfReader

from medagent.core.config import get_settings
from medagent.infra.context_budget import truncate_text
from medagent.rag.chunker import CHUNK_OVERLAP_TOKENS, Chunk, chunk_text
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.pipeline import IngestionPipeline
from medagent.rag.retriever import GuidelineRetrieverTool
from medagent.rag.vector_store import VectorStore
from tests.eval.retrieval_golden_dataset import CASES, SOURCES, RetrievalCase

GUIDELINES_DIR = Path(__file__).resolve().parents[2] / "guidelines_pdf"
TOP_K = 5
AGENT_TOP_K = 3  # agents/evidence_agent.py: search_guidelines


@dataclass(frozen=True)
class Config:
    name: str
    chunk_size: int | None  # None: the embedding model's own limit
    overlap: int
    fit_to_model: bool  # count real tokens (else one word = one token)
    with_context: bool  # embed "<source> › <section>" with each chunk
    drop_references: bool = False  # leave reference-list chunks out
    production: bool = False  # ingest through IngestionPipeline exactly as the app does


CONFIGS = [
    Config("1 baseline: 512 words", 512, 64, fit_to_model=False, with_context=False),
    Config("2 fit to model (254 tokens)", None, CHUNK_OVERLAP_TOKENS, True, False),
    Config("3 + context prefix", None, CHUNK_OVERLAP_TOKENS, True, True),
    Config(
        "4 + no reference lists (production)",
        None,
        CHUNK_OVERLAP_TOKENS,
        True,
        True,
        drop_references=True,
        production=True,
    ),
]


@dataclass
class Report:
    config: Config
    chunks: int
    beyond_limit: float
    ranks: dict[str, int | None] = field(default_factory=dict)  # case id -> 1-based rank
    reached: dict[str, bool] = field(default_factory=dict)  # case id -> answer reached the LLM

    def recall(self, k: int) -> float:
        return sum(1 for r in self.ranks.values() if r is not None and r <= k) / len(self.ranks)

    @property
    def mrr(self) -> float:
        return sum(1 / r for r in self.ranks.values() if r is not None) / len(self.ranks)

    @property
    def reaches_model(self) -> float:
        return sum(self.reached.values()) / len(self.reached)

    @property
    def misses(self) -> list[str]:
        return [case for case, rank in self.ranks.items() if rank is None]


def available() -> bool:
    return all((GUIDELINES_DIR / name).exists() for name in SOURCES.values())


def _normalise(text: str) -> str:
    return " ".join(text.split())


def load_texts() -> dict[str, str]:
    texts = {}
    for source, name in SOURCES.items():
        reader = PdfReader(GUIDELINES_DIR / name)
        texts[source] = "\n".join(page.extract_text() or "" for page in reader.pages)
    return texts


def _chunks(config: Config, model: EmbeddingModel, text: str, source: str) -> list[Chunk]:
    return chunk_text(
        text,
        config.chunk_size or model.max_tokens,
        config.overlap,
        count_tokens=model.count_tokens if config.fit_to_model else None,
        title=source if config.with_context else None,
        drop_reference_lists=config.drop_references,
    )


async def _build_index(
    config: Config, model: EmbeddingModel, texts: dict[str, str]
) -> tuple[VectorStore, int, float]:
    store = VectorStore("unused", 0, f"eval-{uuid.uuid4().hex}", client=chromadb.EphemeralClient())
    total = beyond = count = 0
    for source, text in texts.items():
        chunks = _chunks(config, model, text, source)
        for chunk in chunks:
            tokens = sum(model.count_tokens(chunk.embedding_text.split()))
            total += tokens
            beyond += max(0, tokens - model.max_tokens)
        count += len(chunks)
        if config.production:
            await IngestionPipeline(model, store).ingest_text(text, source)
        else:
            vectors = await model.embed([c.embedding_text for c in chunks])
            await store.add(
                ids=[str(uuid.uuid4()) for _ in chunks],
                documents=[c.text for c in chunks],
                embeddings=vectors,
                metadatas=[{"source": source, "section": c.section or ""} for c in chunks],
            )
    return store, count, beyond / max(1, total)


async def _score(retriever: GuidelineRetrieverTool, case: RetrievalCase) -> tuple[int | None, bool]:
    """(rank of the first passage containing the answer, whether the answer
    survives into the text the evidence agent gives the LLM)."""
    hits = await retriever.run(query=case.query, top_k=TOP_K)
    phrase = _normalise(case.expected_phrase)
    rank = next(
        (i for i, hit in enumerate(hits, start=1) if phrase in _normalise(hit.summary)), None
    )
    # Mirrors search_guidelines + the ReAct loop's cap on tool output.
    tool_output = "\n\n".join(
        f"[{h.source or 'unknown'}] {h.title}:\n{h.summary}" for h in hits[:AGENT_TOP_K]
    )
    seen = truncate_text(
        tool_output, get_settings().agent_context_field_max_tokens, source="retrieval_eval"
    )
    return rank, phrase in _normalise(seen)


async def evaluate(
    configs: list[Config] = CONFIGS, model: EmbeddingModel | None = None
) -> list[Report]:
    model = model or EmbeddingModel()
    texts = load_texts()
    reports = []
    for config in configs:
        store, count, beyond = await _build_index(config, model, texts)
        retriever = GuidelineRetrieverTool(model, store)
        report = Report(config, count, beyond)
        for case in CASES:
            report.ranks[case.id], report.reached[case.id] = await _score(retriever, case)
        reports.append(report)
    return reports


def format_reports(reports: list[Report]) -> str:
    header = (
        f"{'configuration':36} {'chunks':>6} {'beyond limit':>12} "
        f"{'recall@1':>8} {'recall@3':>8} {'recall@5':>8} {'MRR':>6} {'reaches model':>13}"
    )
    lines = [f"Retrieval eval: {len(CASES)} questions over {len(SOURCES)} guidelines", header]
    lines.append("-" * len(header))
    for r in reports:
        lines.append(
            f"{r.config.name:36} {r.chunks:>6} {r.beyond_limit:>12.0%} "
            f"{r.recall(1):>8.0%} {r.recall(3):>8.0%} {r.recall(5):>8.0%} {r.mrr:>6.2f} "
            f"{r.reaches_model:>13.0%}"
        )
    for r in reports:
        if r.misses:
            lines.append(f"\nnot in top {TOP_K} -- {r.config.name}: {', '.join(r.misses)}")
    return "\n".join(lines)


if __name__ == "__main__":
    if not available():
        raise SystemExit(
            f"Guideline PDFs missing from {GUIDELINES_DIR}: {sorted(SOURCES.values())}"
        )
    print(format_reports(asyncio.run(evaluate())))
