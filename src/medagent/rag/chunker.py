"""Section-header-aware chunker, sized to the embedding model's token limit.

A chunk is useless past the embedding model's limit: all-MiniLM-L6-v2 reads 256
tokens and silently ignores the rest. Chunking 512 *words* (about 760 tokens of
guideline text) left two thirds of every chunk unembedded, so search could not
find text in it. Pass the model's own token counter (EmbeddingModel.count_tokens)
and its limit, and every chunk -- context prefix included -- fits.

Each chunk can carry a context prefix ("<document> › <section>") that is
embedded with it but not stored as its text: a passage about "target <120 mm Hg"
is found more reliably when the embedding also says it is the KDIGO blood
pressure section.

Reference lists can be dropped (drop_reference_lists): a guideline's
bibliography is extracted as ordinary text, and citation titles ("Risk-based
triage for nephrology referrals using the kidney failure risk equation") match
questions better than the recommendation that answers them.

Without a token counter, one word counts as one token (rough, but dependency-free).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

# What counts as a section header, in guideline PDFs as pypdf extracts them:
#   - markdown "# Title" lines;
#   - numbered headings such as "2.2 Risk prediction" (needs at least one dot, so a
#     reference-list "1. Smith" or a lone number is not one; the title starts with a
#     capital, so a dose like "2.5 mg" is not one, and it must not end like a sentence);
#   - ALL-CAPS titles of two or more real words ("INTRODUCTION AND METHODOLOGY").
# Table cells and stray labels ("CKD G1", "NA/NA", "A C A C", a lone "CITATION")
# used to match an older all-caps rule and became junk section names.
_HEADER_PATTERN = re.compile(
    r"^(?:"
    r"#{1,6}[ \t]+.*"
    r"|\d+(?:\.\d+)+[ \t]+[A-Z][^\n]{2,90}(?<![.:;,])"
    r"|[A-Z]{3,}(?:[ \t]+[A-Z]{3,})+:?"
    r")$",
    re.MULTILINE,
)

# all-MiniLM-L6-v2: 256 tokens minus [CLS] and [SEP]. The pipeline passes the
# loaded model's own limit; these defaults only apply when no model is involved.
CHUNK_SIZE_TOKENS = 254
# Bump when chunking changes what a document becomes: ingestion re-chunks and
# re-embeds every document whose stored version has another chunker version.
CHUNKER_VERSION = "2026-11-14"
CHUNK_OVERLAP_TOKENS = 32

TokenCounter = Callable[[list[str]], list[int]]

# A journal citation ("2023;388:117-127", "2022;33:301") or "et al.". Guideline
# body text cites by superscript number, so these cluster in reference lists:
# measured on KDIGO 2024, chunks have either none or 3+ per 100 words.
_CITATION = re.compile(r"\b(?:19|20)\d{2};\s?\d+(?:\(\d+\))?:\s?[A-Za-z]?\d+|\bet al\.")
_REFERENCES_PER_100_WORDS = 2.0
_MIN_CITATIONS = 3


def is_reference_list(text: str) -> bool:
    citations = len(_CITATION.findall(text))
    words = len(text.split())
    return (
        citations >= _MIN_CITATIONS and citations * 100 / max(1, words) >= _REFERENCES_PER_100_WORDS
    )


@dataclass
class Chunk:
    text: str
    section: str | None
    # "<document> › <section>": embedded together with the text, not stored as it.
    context: str | None = None

    @property
    def embedding_text(self) -> str:
        return f"{self.context}\n{self.text}" if self.context else self.text


def _one_token_per_word(words: list[str]) -> list[int]:
    return [1] * len(words)


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    matches = list(_HEADER_PATTERN.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append((None, text[: matches[0].start()]))

    for i, match in enumerate(matches):
        header = match.group().strip("# ").strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((header, text[match.end() : end]))

    return sections


def chunk_text(
    text: str,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    *,
    count_tokens: TokenCounter | None = None,
    title: str | None = None,
    drop_reference_lists: bool = False,
) -> list[Chunk]:
    """Splits text into sections, then each section into windows of at most
    `chunk_size_tokens` tokens -- context prefix included -- with about
    `overlap_tokens` tokens repeated between consecutive windows.

    `count_tokens` maps words to their token counts (one batched call per
    section); a word-piece tokenizer splits on whitespace first, so per-word
    counts add up to the count of the joined text. `title` (the document name)
    becomes the first part of each chunk's context prefix. With
    `drop_reference_lists`, windows that are mostly citations are left out."""
    count = count_tokens or _one_token_per_word
    chunks: list[Chunk] = []
    for section, body in _split_into_sections(text):
        words = body.split()
        if not words:
            continue

        context = " › ".join(part for part in (title, section) if part) or None
        budget = chunk_size_tokens - (sum(count(context.split())) if context else 0)
        # A context longer than the budget still leaves room for one word.
        budget = max(budget, 1)
        costs = count(words)

        start = 0
        while start < len(words):
            end, used = start, 0
            # Always take at least one word, so an over-long "word" (a URL, a
            # run-together table row) becomes its own chunk instead of looping.
            while end < len(words) and (end == start or used + costs[end] <= budget):
                used += costs[end]
                end += 1
            window = " ".join(words[start:end])
            if not (drop_reference_lists and is_reference_list(window)):
                chunks.append(Chunk(text=window, section=section, context=context))
            if end >= len(words):
                break
            # Step back by up to overlap_tokens, but always move forward.
            next_start, overlap = end, 0
            while next_start > start + 1 and overlap + costs[next_start - 1] <= overlap_tokens:
                next_start -= 1
                overlap += costs[next_start]
            start = next_start

    return chunks
