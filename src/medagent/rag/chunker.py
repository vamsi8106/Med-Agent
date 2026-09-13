"""Section-header-aware chunker: splits guideline text into ~512-token windows.

Token counts are approximated by whitespace-separated word counts, which avoids
pulling in a tokenizer dependency purely for chunk sizing.
"""

import re
from dataclasses import dataclass

_HEADER_PATTERN = re.compile(r"^(#{1,6}\s+.*|[A-Z][A-Z0-9 /-]{3,}:?)$", re.MULTILINE)

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64


@dataclass
class Chunk:
    text: str
    section: str | None


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
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in _split_into_sections(text):
        words = body.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = start + chunk_size_tokens
            window = words[start:end]
            chunks.append(Chunk(text=" ".join(window), section=section))
            if end >= len(words):
                break
            start = end - overlap_tokens

    return chunks
