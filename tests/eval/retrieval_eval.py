"""Retrieval eval: how often does guideline search put the right passage on top?

Builds a pgvector index from the local guideline PDFs per chunking
configuration (each in its own database of a throwaway Postgres container),
asks every question in retrieval_golden_dataset.py, and reports recall@k and MRR.
Run `make rag-eval` before and after any change to chunking, the embedding
model or retrieval, and compare.

- recall@k: share of questions whose answer passage is in the top k results.
- MRR (mean reciprocal rank): average of 1/rank of the first correct result
  (1.0 = always first; 0 when it is not in the top 5).
- reaches model: share of questions whose answer passage survives into what
  the evidence agent actually hands the LLM -- the top 3 passages, joined and
  capped at AGENT_CONTEXT_FIELD_MAX_TOKENS exactly as `search_guidelines` does.
  Recall@k alone flatters big chunks: 5 chunks of 512 words are ~2,300 words,
  most of which the agent cuts before the model sees it.
- ms/query: retrieval latency on this machine (embedding, search, re-ranking).
- beyond limit: share of embedded tokens past the model's limit, which the
  model silently ignores.

Configurations 1-4 compare chunking; 5-7 compare ranking on chunking 4:
re-ranking with a cross-encoder (rag/reranker.py, RAG_RERANKER_MODEL) and hybrid
keyword + vector search (Postgres full-text search fused by RRF, RAG_HYBRID) --
the exact retriever production runs.

Needs the PDFs in guidelines_pdf/ (gitignored), Docker (for the throwaway
pgvector Postgres), the local embedding model and, for 5 and 7, the re-ranker
model (downloaded from Hugging Face on first use).
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg
from pypdf import PdfReader

from medagent.core.config import get_settings
from medagent.infra.context_budget import truncate_text
from medagent.memory.persistent import PersistentStore
from medagent.memory.schema import ADD_GUIDELINE_VECTOR_STATEMENTS
from medagent.rag.chunker import CHUNK_OVERLAP_TOKENS, CHUNKER_VERSION, Chunk, chunk_text
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.pipeline import IngestionPipeline, chunk_id
from medagent.rag.reranker import CrossEncoderReranker
from medagent.rag.retriever import GuidelineRetrieverTool
from medagent.rag.vector_store import StoredChunk, VectorStore
from tests.eval.retrieval_golden_dataset import CASES, SOURCES, RetrievalCase

GUIDELINES_DIR = Path(__file__).resolve().parents[2] / "guidelines_pdf"
TOP_K = 5
AGENT_TOP_K = 3  # agents/evidence_agent.py: search_guidelines
CANDIDATES = 20  # first-stage results handed to a re-ranker / fused in hybrid

GENERAL_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"  # Apache-2.0


@dataclass(frozen=True)
class Chunking:
    chunk_size: int | None  # None: the embedding model's own limit
    overlap: int
    fit_to_model: bool  # count real tokens (else one word = one token)
    with_context: bool  # embed "<source> › <section>" with each chunk
    drop_references: bool = False  # leave reference-list chunks out


BASELINE = Chunking(512, 64, fit_to_model=False, with_context=False)
FITTED = Chunking(None, CHUNK_OVERLAP_TOKENS, fit_to_model=True, with_context=False)
WITH_CONTEXT = Chunking(None, CHUNK_OVERLAP_TOKENS, fit_to_model=True, with_context=True)
PRODUCTION_CHUNKING = Chunking(None, CHUNK_OVERLAP_TOKENS, True, True, drop_references=True)


@dataclass(frozen=True)
class Config:
    name: str
    chunking: Chunking
    reranker: str | None = None
    hybrid: bool = False
    production: bool = False  # what the app runs with the default settings


CONFIGS = [
    Config("1 baseline: 512 words", BASELINE),
    Config("2 fit to model (254 tokens)", FITTED),
    Config("3 + context prefix", WITH_CONTEXT),
    Config("4 + no reference lists", PRODUCTION_CHUNKING, production=True),
    Config("5 4 + re-rank (MiniLM)", PRODUCTION_CHUNKING, reranker=GENERAL_RERANKER),
    Config("6 4 + hybrid (full-text + vector)", PRODUCTION_CHUNKING, hybrid=True),
    Config("7 6 + re-rank (MiniLM)", PRODUCTION_CHUNKING, GENERAL_RERANKER, hybrid=True),
]


@dataclass
class Index:
    store: VectorStore
    persistent: PersistentStore
    chunks: int
    beyond_limit: float


@dataclass
class Report:
    config: Config
    chunks: int
    beyond_limit: float
    ranks: dict[str, int | None] = field(default_factory=dict)  # case id -> 1-based rank
    reached: dict[str, bool] = field(default_factory=dict)  # case id -> answer reached the LLM
    seconds: list[float] = field(default_factory=list)  # retrieval latency per question

    def recall(self, k: int) -> float:
        return sum(1 for r in self.ranks.values() if r is not None and r <= k) / len(self.ranks)

    @property
    def mrr(self) -> float:
        return sum(1 / r for r in self.ranks.values() if r is not None) / len(self.ranks)

    @property
    def reaches_model(self) -> float:
        return sum(self.reached.values()) / len(self.reached)

    @property
    def ms_per_query(self) -> float:
        return 1000 * sum(self.seconds) / len(self.seconds)

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


def _chunks(chunking: Chunking, model: EmbeddingModel, text: str, source: str) -> list[Chunk]:
    return chunk_text(
        text,
        chunking.chunk_size or model.max_tokens,
        chunking.overlap,
        count_tokens=model.count_tokens if chunking.fit_to_model else None,
        title=source if chunking.with_context else None,
        drop_reference_lists=chunking.drop_references,
    )


def _database_dsn(dsn: str, name: str) -> str:
    return urlunsplit(urlsplit(dsn)._replace(path=f"/{name}"))


async def _fresh_database(dsn: str) -> str:
    """A new database in the same server with the guideline tables, so each
    chunking configuration is indexed on its own."""
    name = f"rag_eval_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(dsn)
    try:
        await admin.execute(f"CREATE DATABASE {name}")
    finally:
        await admin.close()
    database = _database_dsn(dsn, name)
    conn = await asyncpg.connect(database)
    try:
        for statement in ADD_GUIDELINE_VECTOR_STATEMENTS:
            await conn.execute(statement)
    finally:
        await conn.close()
    return database


async def _build_index(
    chunking: Chunking, model: EmbeddingModel, texts: dict[str, str], dsn: str
) -> Index:
    persistent = PersistentStore(await _fresh_database(dsn))
    store = VectorStore(persistent)
    count = total = beyond = 0
    for source, text in texts.items():
        chunks = _chunks(chunking, model, text, source)
        for chunk in chunks:
            tokens = sum(model.count_tokens(chunk.embedding_text.split()))
            total += tokens
            beyond += max(0, tokens - model.max_tokens)
        count += len(chunks)
        if chunking == PRODUCTION_CHUNKING:
            # Exactly the app's ingestion path.
            await IngestionPipeline(model, store).ingest_text(text, source)
        else:
            vectors = await model.embed([c.embedding_text for c in chunks])
            await store.replace_document(
                source,
                content_hash=source,
                embedding_model=model.model_name,
                chunker_version=CHUNKER_VERSION,
                chunks=[
                    StoredChunk(chunk_id(source, i, c.text), i, c.section, c.text, vector)
                    for i, (c, vector) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )
    return Index(store, persistent, count, beyond / max(1, total))


async def _score(
    retriever: GuidelineRetrieverTool, case: RetrievalCase
) -> tuple[int | None, bool, float]:
    """(rank of the first passage containing the answer, whether the answer
    survives into the text the evidence agent gives the LLM, seconds taken)."""
    started = time.perf_counter()
    hits = await retriever.run(query=case.query, top_k=TOP_K)
    elapsed = time.perf_counter() - started
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
    return rank, phrase in _normalise(seen), elapsed


async def evaluate(
    dsn: str, configs: list[Config] = CONFIGS, model: EmbeddingModel | None = None
) -> list[Report]:
    """`dsn`: any Postgres with pgvector where the user may create databases
    (the eval never touches existing data)."""
    model = model or EmbeddingModel()
    texts = load_texts()
    indexes: dict[Chunking, Index] = {}
    rerankers: dict[str, CrossEncoderReranker] = {}
    reports = []
    try:
        for config in configs:
            if config.chunking not in indexes:
                indexes[config.chunking] = await _build_index(config.chunking, model, texts, dsn)
            index = indexes[config.chunking]
            reranker = None
            if config.reranker:
                reranker = rerankers.setdefault(
                    config.reranker, CrossEncoderReranker(config.reranker)
                )
            retriever = GuidelineRetrieverTool(
                model, index.store, reranker=reranker, candidates=CANDIDATES, hybrid=config.hybrid
            )
            await retriever.run(query="warm-up", top_k=1)  # first call loads lazily
            report = Report(config, index.chunks, index.beyond_limit)
            for case in CASES:
                rank, reached, seconds = await _score(retriever, case)
                report.ranks[case.id], report.reached[case.id] = rank, reached
                report.seconds.append(seconds)
            reports.append(report)
    finally:
        for index in indexes.values():
            await index.persistent.close()
    return reports


def format_reports(reports: list[Report]) -> str:
    header = (
        f"{'configuration':34} {'chunks':>6} {'beyond':>6} {'recall@1':>8} {'recall@3':>8} "
        f"{'recall@5':>8} {'MRR':>5} {'reaches model':>13} {'ms/query':>8}"
    )
    lines = [f"Retrieval eval: {len(CASES)} questions over {len(SOURCES)} guidelines", header]
    lines.append("-" * len(header))
    for r in reports:
        lines.append(
            f"{r.config.name:34} {r.chunks:>6} {r.beyond_limit:>6.0%} {r.recall(1):>8.0%} "
            f"{r.recall(3):>8.0%} {r.recall(5):>8.0%} {r.mrr:>5.2f} {r.reaches_model:>13.0%} "
            f"{r.ms_per_query:>8.0f}"
        )
    for r in reports:
        if r.misses:
            lines.append(f"\nnot in top {TOP_K} -- {r.config.name}: {', '.join(r.misses)}")
    return "\n".join(lines)


if __name__ == "__main__":
    from tests.conftest import throwaway_postgres

    if not available():
        raise SystemExit(
            f"Guideline PDFs missing from {GUIDELINES_DIR}: {sorted(SOURCES.values())}"
        )
    with throwaway_postgres() as postgres:
        if postgres is None:
            raise SystemExit("Docker is needed for the throwaway pgvector Postgres")
        print(format_reports(asyncio.run(evaluate(postgres))))
