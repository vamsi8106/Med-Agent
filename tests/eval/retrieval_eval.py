"""Retrieval eval: how often does guideline search put the right passage on top?

Builds an in-memory ChromaDB index from the local guideline PDFs per chunking
configuration, asks every question in retrieval_golden_dataset.py, and reports
recall@k and MRR. Run `make rag-eval` before and after any change to chunking,
the embedding model or retrieval, and compare.

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

Configurations 1-4 compare chunking; 5-8 compare ranking on chunking 4:
re-ranking with a cross-encoder (rag/reranker.py, what production can switch
on) and hybrid keyword + vector search. Hybrid uses an in-memory BM25 here to
measure whether it is worth building (in production it needs a keyword index,
e.g. Postgres full-text search with pgvector).

Needs the PDFs in guidelines_pdf/ (gitignored), the local embedding model and,
for 5-8, the re-ranker models (downloaded from Hugging Face on first use).
"""

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import chromadb
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from medagent.core.config import get_settings
from medagent.core.models import ClinicalEvidence
from medagent.infra.context_budget import truncate_text
from medagent.rag.chunker import CHUNK_OVERLAP_TOKENS, Chunk, chunk_text
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.pipeline import IngestionPipeline
from medagent.rag.reranker import CrossEncoderReranker
from medagent.rag.retriever import GuidelineRetrieverTool
from medagent.rag.vector_store import VectorStore
from tests.eval.retrieval_golden_dataset import CASES, SOURCES, RetrievalCase

GUIDELINES_DIR = Path(__file__).resolve().parents[2] / "guidelines_pdf"
TOP_K = 5
AGENT_TOP_K = 3  # agents/evidence_agent.py: search_guidelines
CANDIDATES = 20  # first-stage results handed to a re-ranker / fused in hybrid
RRF_K = 60  # reciprocal rank fusion constant (the usual default)

GENERAL_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"  # Apache-2.0
MEDICAL_RERANKER = "ncbi/MedCPT-Cross-Encoder"  # public domain (NIH/NCBI)


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
    Config("5 4 + re-rank (general MiniLM)", PRODUCTION_CHUNKING, reranker=GENERAL_RERANKER),
    Config("6 4 + re-rank (medical MedCPT)", PRODUCTION_CHUNKING, reranker=MEDICAL_RERANKER),
    Config("7 4 + hybrid BM25 + vector", PRODUCTION_CHUNKING, hybrid=True),
    Config("8 7 + re-rank (MedCPT)", PRODUCTION_CHUNKING, MEDICAL_RERANKER, hybrid=True),
]


class Retriever(Protocol):
    async def run(self, query: str, top_k: int = 5) -> list[ClinicalEvidence]: ...


@dataclass
class Index:
    store: VectorStore
    chunks: list[tuple[str, Chunk]]  # (source, chunk), the indexed corpus
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


async def _build_index(chunking: Chunking, model: EmbeddingModel, texts: dict[str, str]) -> Index:
    store = VectorStore("unused", 0, f"eval-{uuid.uuid4().hex}", client=chromadb.EphemeralClient())
    corpus: list[tuple[str, Chunk]] = []
    total = beyond = 0
    for source, text in texts.items():
        chunks = _chunks(chunking, model, text, source)
        for chunk in chunks:
            tokens = sum(model.count_tokens(chunk.embedding_text.split()))
            total += tokens
            beyond += max(0, tokens - model.max_tokens)
        corpus.extend((source, chunk) for chunk in chunks)
        if chunking == PRODUCTION_CHUNKING:
            # Exactly the app's ingestion path.
            await IngestionPipeline(model, store).ingest_text(text, source)
        else:
            vectors = await model.embed([c.embedding_text for c in chunks])
            await store.add(
                ids=[str(uuid.uuid4()) for _ in chunks],
                documents=[c.text for c in chunks],
                embeddings=vectors,
                metadatas=[{"source": source, "section": c.section or ""} for c in chunks],
            )
    return Index(store, corpus, beyond / max(1, total))


_WORD = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class HybridRetriever:
    """Vector and BM25 keyword results merged by reciprocal rank fusion
    (score = sum of 1 / (RRF_K + rank) over both lists), optionally re-ranked.
    Keyword search catches exact terms -- drug names, "eGFRcr-cys", "SGLT2i" --
    that embeddings blur; RRF needs no score calibration between the two."""

    def __init__(
        self, model: EmbeddingModel, index: Index, reranker: CrossEncoderReranker | None
    ) -> None:
        self._vector = GuidelineRetrieverTool(model, index.store)
        self._corpus = [
            ClinicalEvidence(title=chunk.section or source, summary=chunk.text, source=source)
            for source, chunk in index.chunks
        ]
        self._bm25 = BM25Okapi([_words(chunk.embedding_text) for _, chunk in index.chunks])
        self._reranker = reranker

    async def run(self, query: str, top_k: int = 5) -> list[ClinicalEvidence]:
        vector_hits = await self._vector.run(query=query, top_k=CANDIDATES)
        scores = self._bm25.get_scores(_words(query))
        keyword_hits = [
            self._corpus[i]
            for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CANDIDATES]
        ]
        fused: dict[str, float] = {}
        by_text: dict[str, ClinicalEvidence] = {}
        for hits in (vector_hits, keyword_hits):
            for rank, hit in enumerate(hits, start=1):
                fused[hit.summary] = fused.get(hit.summary, 0.0) + 1 / (RRF_K + rank)
                by_text[hit.summary] = hit
        merged = [by_text[text] for text in sorted(fused, key=lambda t: fused[t], reverse=True)]
        if self._reranker is None:
            return merged[:top_k]
        return await self._reranker.rerank(query, merged[:CANDIDATES], top_k)


async def _score(retriever: Retriever, case: RetrievalCase) -> tuple[int | None, bool, float]:
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
    configs: list[Config] = CONFIGS, model: EmbeddingModel | None = None
) -> list[Report]:
    model = model or EmbeddingModel()
    texts = load_texts()
    indexes: dict[Chunking, Index] = {}
    rerankers: dict[str, CrossEncoderReranker] = {}
    reports = []
    for config in configs:
        if config.chunking not in indexes:
            indexes[config.chunking] = await _build_index(config.chunking, model, texts)
        index = indexes[config.chunking]
        reranker = None
        if config.reranker:
            reranker = rerankers.setdefault(config.reranker, CrossEncoderReranker(config.reranker))
        retriever: Retriever
        if config.hybrid:
            retriever = HybridRetriever(model, index, reranker)
        else:
            retriever = GuidelineRetrieverTool(
                model, index.store, reranker=reranker, candidates=CANDIDATES
            )
        await retriever.run(query="warm-up", top_k=1)  # first call loads lazily
        report = Report(config, len(index.chunks), index.beyond_limit)
        for case in CASES:
            rank, reached, seconds = await _score(retriever, case)
            report.ranks[case.id], report.reached[case.id] = rank, reached
            report.seconds.append(seconds)
        reports.append(report)
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
    if not available():
        raise SystemExit(
            f"Guideline PDFs missing from {GUIDELINES_DIR}: {sorted(SOURCES.values())}"
        )
    print(format_reports(asyncio.run(evaluate())))
