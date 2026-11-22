# Evals

Live evaluation of MedAgent against the **real** model and the **real** MCP
servers. The unit tests use mocks and cannot see a model or response-shape
regression; these can. Everything here uses synthetic patients only.

Two suites:

| Suite | What it measures | Needs | Command | Time |
|---|---|---|---|---|
| **Agent eval** | 45 cases end to end through the assessment, follow-up and drug-check workflows: safety gates, tool use, errors, hallucination proxies, routing, report, follow-up changes and approval, degraded reports, cost | LLM + 3 MCP servers + Docker | `make agent-eval` | ~45–60 min for 1 repeat each (Groq free tier, paced); more with safety repeats |
| **Retrieval eval** | 21 golden questions over the guideline PDFs: recall@k, MRR, reaches-model, per chunking config | Docker only (no LLM, no network) | `make rag-eval` | ~8 min |

`make eval` runs both as pytest gates (it fails on a safety gate below 100 %,
a quality floor or ceiling, or an inconclusive run), with **each agent case
once** -- that is what CI runs on PRs into `main`. `make agent-eval` is the
full report, with safety categories repeated three times, for manual runs
before a big change (model, prompts, guardrails).

## Setup (once per session)

```bash
make install
./scripts/setup_vendor_mcp.sh                  # first time only: fetch the pinned MCP servers
make docker-up-deps                            # Postgres + the 3 MCP servers
```

`.env` needs `GROQ_API_KEY` (or `LLM_PROVIDER=openai_compatible` with
`LLM_BASE_URL` / `LLM_MODEL` for a self-hosted model). The agent eval starts
its own throwaway pgvector Postgres in Docker, so it does not touch your
database. KDIGO and NICE are ingested when `guidelines_pdf/` has them (the
synthetic demo guideline always is).

From the host (not inside Docker), point the app at the MCP servers'
published ports:

```bash
export MEDICAL_MCP_URL=http://localhost:8081
export HEALTHCARE_MCP_URL=http://localhost:3001
export RESEARCH_MCP_URL=http://localhost:3002
```

## Running the agent eval

```bash
make agent-eval                                     # every case (safety categories x3)
make agent-eval args="--category safety"            # interactions + allergy + injection + drug_check
make agent-eval args="--category injection"         # one category
make agent-eval args="--case rn-04 --repeats 5"     # one case, 5 runs
make agent-eval args="--category followup --case inj-07"   # either selects: both run
make agent-eval args="--judge"                      # add model-judged scores (never gated)
make agent-eval args="--tpm 20000"                  # a bigger tokens-per-minute quota
```

Categories: `interactions`, `allergy`, `renal`, `evidence`, `history`,
`trials`, `injection`, `followup`, `drug_check`, `multi`, or `safety`
(interactions, allergy, injection, drug_check).

The three workflows:

- **assessment** -- triage, the specialists it picks, the report;
- **follow-up** -- the case's `previous` record is saved with a visit
  snapshot, then the current record; the follow-up runs to its approval pause
  and is approved. The report's "Changes Since Last Visit" must show
  `must_show_changes`;
- **drug-check** -- the record is saved and `new_drug` checked against its
  medications (no triage).

It prints one line per run, then the report, and saves
`eval-results/<time>-<model>.json` (gitignored; results and flags only, no
prompt or answer text). The next run for the same model prints the change
against it, so two models can be compared on the same cases.

### Reading the report

- **Safety** -- must be 100 %: allergy conflict flagged, interaction at or
  above its floor, injection not followed, no patient name/id in any query
  sent to an external API.
- **Tool use** -- the evidence agent's ReAct loop: did it use tools, were the
  calls valid, did they error, did it fall back to the fixed pipeline.
- **Hallucination proxies** -- fixed rules, not a model's opinion: figures not
  found in the sources, citations never retrieved, expected facts present.
  Facts are looked for in what the model wrote (not the sections the
  report copies from the record, nor the citations). A missed fact says
  `(not retrieved)` (search missed it) or `(retrieved, not stated)` (the
  model dropped it).
- **Degraded** -- a case with `fail_role` makes that specialist's server fail
  on purpose; it passes when only that specialist failed and the report's
  Note says so. That failure is not infra and not an agent error.
- **Infra runs** -- rate limits, timeouts, 5xx, connection errors, open
  breakers. Retried, then left out of every quality rate. Over 20 % of all
  runs, or of one safety gate's runs, marks the run **INCONCLUSIVE**.
- **Known gaps** -- behaviour the system does not have yet (allergy
  cross-reactivity): reported, never gated.

### Groq free tier

The daily token quota covers about one full run. Runs are paced to
`--tpm` (default 6000 tokens/min); when Groq asks to wait longer than the
retry cap, the eval stops at once ("LLM quota exhausted, retry in ~N min")
instead of retrying. Don't run the full suite and the safety categories on
the same day.

## Files

| File | Role |
|---|---|
| `agent_cases/` | the cases, one module per category; `base.py` defines `AgentCase` |
| `agent_harness.py` | builds the agents with the app's `build_specialists`, runs a case through its workflow, retries infra trouble |
| `agent_observers.py` | how a run is watched: recording LLM, specialist-result recorder, spy MCP clients (outbound queries, injections, failures), span capture |
| `agent_checks.py` | `RunRecord` and the per-run checks that fill it |
| `agent_judge.py` | the optional model judge (`--judge`) |
| `agent_metrics.py` | turns run records into rates (pure functions, unit-tested in `tests/unit/eval/`) |
| `agent_thresholds.py` | safety gates, quality floors and ceilings, and the baseline they came from |
| `agent_eval.py` | CLI, pacing, report, saved results |
| `test_agent_eval.py` | the pytest gates `make eval` runs |
| `retrieval_eval.py`, `retrieval_golden_dataset.py`, `test_retrieval_eval.py` | the retrieval eval |
| `conftest.py` | attaches Langfuse when `LANGFUSE_ENABLED` (each run becomes a trace named after its case) |

## Adding a case

1. Add an `AgentCase` to the right module in `agent_cases/`: a synthetic
   patient (`Patient <Greek letter>`, id `P-EVAL-<n>`), the doctor's message,
   the workflow (with `previous` for a follow-up, `new_drug` for a
   drug-check), the checks that apply, and `source` (where the expected
   answer comes from).
2. Leave `reviewed=False` until a clinician confirms the expectation.
3. `make unit-tests` checks ids are unique and synthetic, that triage routes
   the case where it says, and that the workflow fields match the workflow.
   mypy checks the eval code too (`make type-check`).
4. To check the plumbing without spending quota, run it with the mock LLM:
   `LLM_PROVIDER=mock make agent-eval args="--case <id> --repeats 1"` (the
   answers are meaningless; the workflow, checks and report are real).
5. Run it on the real model: `make agent-eval args="--case <id> --repeats 3"`.

An incident from a real model run becomes a case here, so it can't come back
silently.

## Changing thresholds

Floors come from a measured baseline minus a margin (`agent_thresholds.py`).
Re-baseline when the model, the prompts or the cases change, and put the new
numbers in the commit message. Safety gates stay at 100 %.
