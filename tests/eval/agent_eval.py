"""Agent eval: run the cases end to end and report the rates.

    make agent-eval                                   # every case
    make agent-eval args="--category safety"          # interactions, allergy, injection
    make agent-eval args="--case rn-02 --repeats 5 --judge"

Needs the LLM (GROQ_API_KEY, or LLM_PROVIDER=openai_compatible), the three MCP
servers (`make docker-up-deps`; from the host set MEDICAL_MCP_URL,
HEALTHCARE_MCP_URL and RESEARCH_MCP_URL to the localhost ports), Docker for a
throwaway pgvector Postgres, and the local embedding model. KDIGO and NICE are
ingested when guidelines_pdf/ has them.

Prints the rates and every failing case, saves eval-results/<time>-<model>.json
(results and flags only, no prompt or answer text) and compares it with the
latest earlier file for the same model. With LANGFUSE_ENABLED each run is a
Langfuse trace named after its case, in one session per eval run.
"""

import argparse
import asyncio
import json
import re
import subprocess
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medagent.core.config import get_settings
from tests.eval import retrieval_eval
from tests.eval.agent_cases import ALL_CASES, SAFETY_CATEGORIES, AgentCase
from tests.eval.agent_checks import RunRecord
from tests.eval.agent_harness import EVAL_RUN_ID, Suite, prepared_suite, run_case
from tests.eval.agent_metrics import Rate, Summary, summarize

RESULTS_DIR = Path(__file__).resolve().parents[2] / "eval-results"
DEFAULT_TOKENS_PER_MINUTE = 6000  # under Groq free tier's 8000 TPM for gpt-oss-20b


class TokenPacer:
    """Keeps the eval under a tokens-per-minute quota: before each run, waits
    until the tokens spent in the last 60 s are below `tokens_per_minute`."""

    def __init__(
        self,
        tokens_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.tokens_per_minute = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._spent: deque[tuple[float, int]] = deque()

    def spent(self, tokens: int) -> None:
        self._spent.append((self._clock(), tokens))

    def _in_window(self) -> int:
        now = self._clock()
        while self._spent and now - self._spent[0][0] >= 60:
            self._spent.popleft()
        return sum(tokens for _, tokens in self._spent)

    async def wait(self) -> float:
        """Sleeps as needed; returns the seconds waited."""
        waited = 0.0
        while self._spent and self._in_window() >= self.tokens_per_minute:
            pause = max(0.5, 60 - (self._clock() - self._spent[0][0]))
            await self._sleep(pause)
            waited += pause
        return waited


def default_repeats(case: AgentCase) -> int:
    return 3 if case.category in SAFETY_CATEGORIES else 1


def select_cases(categories: list[str], case_ids: list[str]) -> list[AgentCase]:
    wanted = {c for name in categories for c in (SAFETY_CATEGORIES if name == "safety" else {name})}
    if not wanted and not case_ids:
        return list(ALL_CASES)
    # Either selects: "--category followup --case inj-07" runs both.
    return [case for case in ALL_CASES if case.category in wanted or case.id in case_ids]


async def run_cases(
    suite: Suite,
    cases: list[AgentCase],
    repeats: Callable[[AgentCase], int],
    *,
    pacer: TokenPacer | None = None,
    on_run: Callable[[RunRecord], None] | None = None,
) -> list[RunRecord]:
    pacer = pacer or TokenPacer(DEFAULT_TOKENS_PER_MINUTE)
    records = []
    for case in cases:
        for repeat in range(1, repeats(case) + 1):
            waited = await pacer.wait()
            if waited and on_run:
                print(f"  (paced {waited:.0f}s for the token quota)", flush=True)
            record = await run_case(suite, case, repeat)
            pacer.spent(record.tokens)
            records.append(record)
            if on_run:
                on_run(record)
            if record.quota_exhausted:
                if on_run:
                    wait = (record.retry_after or 0) / 60
                    print(f"  LLM quota exhausted (retry in ~{wait:.0f} min): stopping", flush=True)
                return records
    return records


async def evaluate(
    dsn: str,
    cases: list[AgentCase],
    repeats: Callable[[AgentCase], int] = default_repeats,
    *,
    judge: bool = False,
    tokens_per_minute: int = DEFAULT_TOKENS_PER_MINUTE,
    on_run: Callable[[RunRecord], None] | None = None,
) -> tuple[list[RunRecord], Summary, str]:
    texts = retrieval_eval.load_texts() if retrieval_eval.available() else {}
    async with prepared_suite(dsn, guideline_texts=texts, with_judge=judge) as suite:
        records = await run_cases(
            suite, cases, repeats, pacer=TokenPacer(tokens_per_minute), on_run=on_run
        )
        return records, summarize(records, cases), suite.model


# --- report ----------------------------------------------------------------------------

_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Safety (gates: 100 %)",
        [
            ("allergy_flagged", "allergy conflict flagged"),
            ("severity_floor", "interaction at or above its floor"),
            ("injection_resisted", "injection not followed"),
            ("no_identifier_leak", "no identifier in outbound queries"),
        ],
    ),
    (
        "Tool use (evidence ReAct loop)",
        [
            ("tool_use", "tool-use rate"),
            ("tool_call_validity", "tool-call validity"),
            ("tool_error", "tool error rate"),
            ("duplicate_calls", "duplicate / blocked calls"),
            ("expected_tools", "expected tool used"),
            ("fallback", "fallback rate"),
        ],
    ),
    (
        "Hallucination proxies (fixed-rule)",
        [
            ("unsupported_figures", "answers with unsupported figures"),
            ("fabricated_references", "answers citing unretrieved sources"),
            ("expected_fact_recall", "expected-fact recall"),
            ("contradiction", "contradictions"),
        ],
    ),
    (
        "Errors, routing, report",
        [
            ("run_error", "run error rate (agent)"),
            ("infra", "infra-affected runs"),
            ("triage_accuracy", "triage accuracy"),
            ("report_complete", "report complete"),
            ("changes_shown", "follow-up: changes since last visit shown"),
            ("approval_ok", "follow-up: approval step works"),
            ("degraded_ok", "degraded report when a server fails"),
            ("pass", "case pass rate"),
        ],
    ),
]


def _fmt(rate: Rate | None) -> str:
    if rate is None or rate.value is None:
        return "     n/a"
    return f"{rate.value:>7.0%}  ({rate.hits}/{rate.total})"


def format_report(summary: Summary, model: str, seconds: float, previous: dict | None) -> str:
    lines = [
        f"Agent eval  model={model}  runs={summary.runs}  "
        f"infra={summary.infra_runs}  {seconds / 60:.1f} min  session={EVAL_RUN_ID}",
    ]
    if summary.inconclusive:
        lines.append(
            f"!! INCONCLUSIVE: {'; '.join(summary.inconclusive_reasons)} "
            f"({summary.counts.get('infra_reasons')})"
        )
    old = (previous or {}).get("summary", {}).get("rates", {})
    for title, metrics in _SECTIONS:
        lines += ["", title]
        for key, label in metrics:
            rate = summary.rates.get(key)
            line = f"  {label:38} {_fmt(rate)}"
            before = old.get(key, {}).get("value")
            if rate is not None and rate.value is not None and before is not None:
                delta = rate.value - before
                if abs(delta) >= 0.005:
                    line += f"   {'+' if delta > 0 else ''}{delta:.0%} vs last"
            lines.append(line)

    if summary.severity.get("total"):
        s = summary.severity
        lines += [
            "",
            "Interaction severity vs documented (informational)",
            f"  exact {s.get('exact', 0)}  under-graded {s.get('under', 0)}  "
            f"over-graded {s.get('over', 0)}  of {s['total']} runs",
        ]
    cost = summary.cost
    lines += [
        "",
        "Cost and speed per run (median / p95)",
        f"  tokens {cost['tokens']['median']} / {cost['tokens']['p95']}   "
        f"LLM calls {cost['llm_calls']['median']} / {cost['llm_calls']['p95']}   "
        f"seconds {cost['seconds']['median']} / {cost['seconds']['p95']}   "
        f"budget overruns {summary.counts['budget_overruns']['runs']}",
        "",
        "By category",
        *(f"  {cat:14} {_fmt(rate)}" for cat, rate in sorted(summary.by_category.items())),
        "",
        "Counts",
        *(f"  {name}: {values}" for name, values in summary.counts.items() if values),
    ]
    if summary.judge:
        lines += [
            "",
            "Model-judged (1-5, not gated)",
            "  " + "  ".join(f"{k} {v:.1f}" for k, v in summary.judge.items() if v is not None),
        ]
    if summary.flaky_cases:
        lines += ["", f"Flaky (passed some repeats): {', '.join(summary.flaky_cases)}"]
    if summary.known_gaps:
        lines += ["", "Known gaps (reported, not gated)"]
        lines += [f"  {case}: {_fmt(rate)}" for case, rate in summary.known_gaps.items()]
    if summary.failures:
        lines += ["", "Failing cases"]
        lines += [f"  {case}: {'; '.join(why)}" for case, why in sorted(summary.failures.items())]
    lines += [
        "",
        f"{summary.unreviewed_cases} case(s) not yet clinician-reviewed "
        "(tests/eval/agent_cases, reviewed=False)",
    ]
    return "\n".join(lines)


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model).strip("_") or "model"


def latest_result(model: str) -> dict | None:
    files = sorted(RESULTS_DIR.glob(f"*-{_slug(model)}.json"))
    return json.loads(files[-1].read_text()) if files else None


def save_result(records: list[RunRecord], summary: Summary, model: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{stamp}-{_slug(model)}.json"
    payload: dict[str, Any] = {
        "run_id": EVAL_RUN_ID,
        "model": model,
        "git_sha": sha,
        "finished_at": stamp,
        "summary": summary.as_json(),
        "records": [r.as_json() for r in records],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _progress(record: RunRecord) -> None:
    status = "infra" if record.infra else ("ok" if not record.raised else record.raised)
    print(f"  {record.case_id:8} #{record.repeat}  {status:6} {record.seconds:6.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--category", action="append", default=[], help="or 'safety'")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--repeats", type=int, help="runs per case (default 3 safety, 1 other)")
    parser.add_argument("--judge", action="store_true", help="add model-judged scores")
    parser.add_argument(
        "--tpm",
        type=int,
        default=DEFAULT_TOKENS_PER_MINUTE,
        help="LLM tokens-per-minute quota to stay under (paces the runs)",
    )
    args = parser.parse_args()

    from tests.conftest import throwaway_postgres

    settings = get_settings()
    if settings.llm_provider == "groq" and not settings.groq_api_key:
        raise SystemExit("GROQ_API_KEY is not set")
    cases = select_cases(args.category, args.case)
    if not cases:
        raise SystemExit("No case matches the selection")
    repeats = (lambda _case: args.repeats) if args.repeats else default_repeats

    from medagent.infra import tracing

    if settings.langfuse_enabled:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            tracing.attach_langfuse(provider, settings)

    started = time.perf_counter()
    with throwaway_postgres() as dsn:
        if dsn is None:
            raise SystemExit("Docker is needed for the throwaway pgvector Postgres")
        records, summary, model = asyncio.run(
            evaluate(
                dsn, cases, repeats, judge=args.judge, tokens_per_minute=args.tpm, on_run=_progress
            )
        )
    previous = latest_result(model)
    path = save_result(records, summary, model)
    tracing.shutdown_tracing()
    print()
    print(format_report(summary, model, time.perf_counter() - started, previous))
    print(f"\nSaved {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
