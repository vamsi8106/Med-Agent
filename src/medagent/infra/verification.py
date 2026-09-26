"""Deterministic check of an answer's numeric claims against its sources.

The costliest hallucination in a clinical answer is an invented number -- a
dose, a percentage, a lab threshold. This finds every figure the answer states
with a unit ("25 mg", "40%", "30 mL/min") and reports those whose number
appears in none of the sources the model was actually given (retrieved
evidence, the patient record, the doctor's question).

Deliberately narrow, and advisory only:
- Figures need a unit. Bare numbers (years, stages, counts, "3 trials") are
  ignored, so the check stays quiet on ordinary prose.
- Only the number is compared, not the unit, and formatting is normalized
  ("1,000" == "1000", "2.0" == "2"), so a faithful answer that reformats a
  figure is not flagged. The cost is that a wrong unit on a real number slips
  through.
- It checks numbers, not whether the reasoning follows from the evidence.
"""

import re

_UNITS = (
    r"(?:%|percent|mg/dl|mg/l|mmol/l|ml/min(?:/1\.73\s?m(?:2|²))?|ng/ml|pg/ml|miu/l|meq/l"
    r"|mmhg|mcg|µg|ug|mg|kg|g|ml|l|mmol|meq|iu|units?|bpm)"
)
_NUMBER = r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?"
_FIGURE = re.compile(
    rf"(?<![\w.])({_NUMBER})(?:\s?[-–]\s?({_NUMBER}))?\s?{_UNITS}(?![a-z])", re.IGNORECASE
)
_BARE_NUMBER = re.compile(rf"(?<![\w.])({_NUMBER})(?!\w)")


def _normalize(number: str) -> str:
    number = number.replace(",", "")
    return number.removesuffix(".0")


def find_unverified_figures(answer: str, *sources: str) -> list[str]:
    """Figures stated in `answer` whose number appears in none of `sources`,
    in order of first appearance, without duplicates."""
    known: set[str] = set()
    for source in sources:
        known.update(_normalize(m.group(1)) for m in _BARE_NUMBER.finditer(source))

    flagged: list[str] = []
    for match in _FIGURE.finditer(answer):
        numbers = [_normalize(n) for n in match.groups() if n]
        shown = match.group(0).strip()
        if any(n not in known for n in numbers) and shown not in flagged:
            flagged.append(shown)
    return flagged


def unverified_figures_warning(figures: list[str]) -> str:
    return (
        f"⚠️ UNVERIFIED FIGURES: the answer states {', '.join(figures)}, which do not appear in "
        "the retrieved evidence or the patient record -- verify before acting."
    )
