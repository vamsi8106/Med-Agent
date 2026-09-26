"""Bounds how much raw external text (MCP/API responses, replayed history)
can enter an LLM prompt. Free medical APIs return payloads of wildly
unpredictable size -- without a cap here, a single drug-interaction analysis
or literature search can inject thousands of tokens of raw JSON into a
prompt, and pairwise interaction checks multiply that by every medication
combination.

Token counts are estimated from character length (3.5 chars/token), not word
count. Measured against real Groq usage on this app's own prompts, dense
medical text runs 3.9-5.0 chars/token -- i.e. 1.6-2.1 tokens per whitespace
word (URLs, DOIs, dosages and citations tokenize heavily). A word-count
estimate undercounted by ~2x and let a prompt sized "6000 tokens" arrive at
Groq as ~9,600, over its 8000-token per-request limit. 3.5 deliberately errs
on the conservative side of the measured range.
"""

import math
import re

from medagent.infra.logging import get_logger
from medagent.infra.metrics import context_truncations_total

logger = get_logger(__name__)

TRUNCATION_MARKER = "\n...[truncated: exceeded context budget]"
_CHARS_PER_TOKEN = 3.5
_TRAILING_PARTIAL_WORD = re.compile(r"\s\S*$")


def approx_token_count(text: str) -> int:
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def truncate_text(text: str, max_tokens: int, *, source: str) -> str:
    """Truncates `text` to approximately `max_tokens` tokens on a word
    boundary, logging when it actually cuts something so context loss is
    observable, not silent."""
    max_chars = int(max_tokens * _CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    if not text[max_chars].isspace():
        partial = _TRAILING_PARTIAL_WORD.search(cut)
        if partial:
            cut = cut[: partial.start()]

    context_truncations_total.labels(source=source).inc()
    logger.warning(
        "context_truncated",
        source=source,
        original_tokens=approx_token_count(text),
        max_tokens=max_tokens,
    )
    return cut + TRUNCATION_MARKER
