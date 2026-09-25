"""Bounds how much raw external text (MCP/API responses) can enter an LLM
prompt. Free medical APIs return payloads of wildly unpredictable size --
without a cap here, a single drug-interaction analysis or literature search
can inject thousands of tokens of raw JSON into a prompt, and pairwise
interaction checks multiply that by every medication combination.

Token counts are approximated by whitespace-separated word count, the same
convention rag/chunker.py already uses, rather than pulling in a real
tokenizer purely for budgeting.
"""

from medagent.infra.logging import get_logger

logger = get_logger(__name__)

TRUNCATION_MARKER = "\n...[truncated: exceeded context budget]"


def approx_token_count(text: str) -> int:
    return len(text.split())


def truncate_text(text: str, max_tokens: int, *, source: str) -> str:
    """Truncates `text` to approximately `max_tokens` words, logging when it
    actually cuts something so context loss is observable, not silent."""
    words = text.split()
    if len(words) <= max_tokens:
        return text

    logger.warning(
        "context_truncated",
        source=source,
        original_tokens=len(words),
        max_tokens=max_tokens,
    )
    return " ".join(words[:max_tokens]) + TRUNCATION_MARKER
