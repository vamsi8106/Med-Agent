# MedAgent — how it works, and why

A guide to the concepts, flows and engineering techniques in this project. It is written so someone new can read it top to bottom and understand the system. Each section says what the thing is, why it exists, and where to find it.

Contents

1. [What MedAgent is](#1-what-medagent-is)
2. [The big picture](#2-the-big-picture)
3. [Layers and dependency rules](#3-layers-and-dependency-rules)
4. [A request, step by step](#4-a-request-step-by-step)
5. [The user flows](#5-the-user-flows)
6. [Agents and the ReAct loop](#6-agents-and-the-react-loop)
7. [Graph engineering (LangGraph)](#7-graph-engineering-langgraph)
8. [Loop engineering](#8-loop-engineering)
9. [Harness engineering: errors, retries, circuit breakers, rate limits](#9-harness-engineering-errors-retries-circuit-breakers-rate-limits)
10. [Context engineering](#10-context-engineering)
11. [Guardrails and verification](#11-guardrails-and-verification)
12. [Human-in-the-loop approval](#12-human-in-the-loop-approval)
13. [RAG: guidelines search](#13-rag-guidelines-search)
14. [Memory and storage](#14-memory-and-storage)
15. [Security, privacy and audit](#15-security-privacy-and-audit)
16. [Observability: logging, metrics, tracing, health](#16-observability-logging-metrics-tracing-health)
17. [Testing strategy](#17-testing-strategy)
18. [Deployment and operations](#18-deployment-and-operations)
19. [Configuration reference](#19-configuration-reference)
20. [Known limitations](#20-known-limitations)
21. [Glossary](#21-glossary)

---

## 1. What MedAgent is

MedAgent is a clinical decision-support assistant for doctors. It:

- checks drug interactions for a patient's medications,
- retrieves treatment evidence from ingested guidelines and PubMed,
- finds clinical trials,
- remembers each patient across visits.

Medical data comes from free, open APIs, reached through three MCP servers (see [section 2](#2-the-big-picture)). Patient records stay on your own infrastructure (Postgres and ChromaDB). Demos and tests use **synthetic patients only** ("Patient Alpha", IDs like `P-TEST-001`).

It is a decision-*support* tool: it drafts reports for a doctor to review, and the WebSocket flow requires the doctor to approve before anything is saved.

**Stack:** Python 3.12, FastAPI, LangGraph, pydantic v2, structlog, httpx, asyncpg + Alembic, ChromaDB, sentence-transformers (`all-MiniLM-L6-v2`, CPU-only), Prometheus, Grafana, OpenTelemetry, Tempo. The LLM is Groq by default, swappable through `LLM_PROVIDER`.

## 2. The big picture

```
                         Doctor (REST or WebSocket, JWT)
                                     │
                       ┌─────────────▼──────────────┐
                       │ FastAPI  (src/medagent/api) │
                       │ middleware: request id,     │
                       │ HTTP span, metrics          │
                       └─────────────┬──────────────┘
                                     │
                       ┌─────────────▼──────────────┐
                       │ LangGraph workflows         │
                       │ init_run → triage →         │
                       │  ┌ drug_safety ┐            │
                       │  ├ evidence    ├ parallel   │
                       │  └ trial_finder┘            │
                       │ → report → (approval)       │
                       └───┬───────────┬─────────────┘
                           │           │
                 ┌─────────▼───┐   ┌───▼─────────────────┐
                 │ Agents       │   │ Memory              │
                 │ (ReAct loop) │   │ Postgres (records)  │
                 └──┬───────┬───┘   │ ChromaDB (RAG)      │
                    │       │       └─────────────────────┘
           ┌────────▼──┐  ┌─▼──────────────────────────────┐
           │ LLM       │  │ Tools                           │
           │ (Groq)    │  │  MCP: medical / healthcare /    │
           │ retry +   │  │       research (over HTTP)      │
           │ breaker   │  │  custom: interaction checker,   │
           └───────────┘  │  lab interpreter, guideline     │
                          │  retriever, report generator    │
                          └─────────────────────────────────┘

Beside all of it:  Prometheus ← /metrics      Tempo ← OTLP traces      Grafana (dashboards)
```

**MCP** (Model Context Protocol) servers give the agents access to medical data. Each runs as its own container and is called over HTTP so they can scale and fail independently:

| Server | Provides | How it is reached |
|---|---|---|
| `medical-mcp` | FDA, WHO, RxNorm, PubMed | stdio-only, so it sits behind a small stdio↔HTTP bridge sidecar |
| `healthcare-mcp` | ICD-10, clinical trials, calculators | its own HTTP API |
| `med-research-mcp-suite` | cross-database analysis, drug-safety profiles | its own REST API |

## 3. Layers and dependency rules

The code is organised in layers. A layer may only import from the layers listed after the arrow. Nothing imports upward.

```
core/      → nothing            shared kernel: models, interfaces, exceptions, config, enums
infra/     → core               logging, retry, rate limit, breaker, metrics, tracing, guardrails
llm/       → core, infra        provider layer (Groq, mock, registry)
tools/     → core, infra        BaseTool, registry, MCP clients, custom tools
memory/    → core               Postgres stores, session buffer
rag/       → core, infra        embeddings, vector store, chunker, pipeline, retriever
auth/      → core, memory       hashing, JWT, user store, FastAPI deps
agents/    → core, infra, memory, rag, tools   (the LLM arrives via an interface, not by importing llm/)
workflows/ → core, infra, agents
api/       → everything except app.py
app.py     → everything
```

Why: it keeps the domain model free of framework code, makes each layer testable in isolation, and stops circular imports. **It is enforced**, not just documented: `make import-check` runs import-linter with one contract per layer, and it runs in `make pre-commit` and in CI.

Other rules the code follows: async for I/O and sync for CPU work, structlog never `print()`, timezone-aware UTC datetimes, `MedAgentError` subclasses instead of bare `Exception`, config only through the `Settings` singleton, and "no abstraction without a second caller".

## 4. A request, step by step

Example: `POST /patients/{id}/assess` with "Check interactions for current medications".

1. **Middleware** (`infra/middleware.py`) accepts or generates a `request_id`, binds it to the log context, starts an HTTP span, and times the request.
2. **Auth** (`auth/dependencies.py`) decodes the JWT and loads the user. No valid token gives 401.
3. **Route** (`api/routes/patients.py`) loads the patient scoped to the doctor (`doctor_scope`). A patient owned by another doctor gives 404, the same as "does not exist".
4. **`run_patient_assessment`** (`workflows/patient_assessment.py`) starts the LangGraph workflow:
   - `init_run` creates the per-run token budget tracker.
   - `triage` chooses which specialists to run (keyword routing; default is the evidence agent).
   - The chosen specialists run **in parallel**: drug safety, evidence, trial finder.
   - `report` merges their output into one markdown report and appends the deterministic sections from the record: Changes Since Last Visit, Known Allergies and Lab Flags.
5. Each specialist runs a ReAct loop, calling tools (MCP servers, guideline search) and the LLM. Every LLM/MCP call passes through retry, a circuit breaker, a rate limiter, and emits metrics and spans.
6. The answer goes through **guardrails** (allergy cross-check, sanity checks, figure verification).
7. The route saves the patient (reconciling medications and labs, never deleting history), saves the visit with a snapshot of the record, writes an **audit-log** entry, and returns `{"report": ...}`.
8. **Middleware** finishes: it records the request metric by route template, logs one `http_request` line, and returns the `X-Request-ID` header.

## 5. The user flows

| Flow | Trigger | What happens |
|---|---|---|
| **New patient** | `POST /patients`, then `/assess` | Save the record. Assessment runs triage, then drug safety and evidence in parallel, then the report. |
| **Follow-up** | `POST /patients/{id}/followup` or the WebSocket | Load the patient and prior visits, compare the record with the last visit's snapshot ("what changed"), run the assessment as a subgraph, then (WebSocket only) wait for approval, then save. |
| **Drug check** | `POST /drug-check` | Load current medications, check every pair through the research MCP, aggregate, and add a deterministic allergy warning if the new drug matches a recorded allergy. |
| **Trial search** | a message mentioning "trial" | The trial-finder agent searches ClinicalTrials.gov through the healthcare MCP. |
| **Evidence lookup** | a clinical question | The evidence agent runs a ReAct loop over PubMed and the guideline store, with citations; on failure it falls back to a fixed pipeline. |

**Triage** is deliberately simple keyword matching (`agents/triage_agent.py`). "interaction/drug/medication/dose" routes to drug safety, "evidence/guideline/literature/treatment" to evidence, "trial/study" to trial finder, and anything else defaults to evidence. It is predictable and free, but it is not semantic. See [limitations](#20-known-limitations).

## 6. Agents and the ReAct loop

An **agent** here is a class that takes a patient context and a message and returns text. The agents are:

| Agent | Job |
|---|---|
| `TriageAgent` | routes the message to specialists (no LLM) |
| `DrugSafetyAgent` | interactions and adverse events, using the interaction checker tool and the LLM |
| `EvidenceAgent` | literature and guideline retrieval, as a ReAct agent with a fixed-pipeline fallback |
| `TrialFinderAgent` | ClinicalTrials.gov search |
| `ReportAgent` | merges results into a markdown report (no LLM) |

**ReAct** (Reason + Act) is a loop: the model thinks, optionally calls a tool, sees the result, and repeats until it can answer. `agents/base.py` implements it as a small LangGraph graph with a `think` node and an `act` node:

```
think ──(tool calls?)──► act ──► think ── ... ──► final answer
```

The evidence agent uses it so the model can choose between `search_guidelines` and `search_medical_literature`, and can refine a search that returned nothing. Its prompt tells it to use broad queries (a condition plus a topic, never a year, a lab value or a full sentence) and to answer only from what the tools returned.

**Fallback:** if the model retrieves nothing, or the loop errors or times out, the evidence agent switches to a fixed "retrieve then summarise" pipeline. A metric records which path was taken, which is the number to watch for the agent's health.

## 7. Graph engineering (LangGraph)

LangGraph models the workflow as a graph of nodes over shared state. What we use it for:

- **Parallel specialists.** Triage decides who runs; they run concurrently and their results merge into the state.
- **Partial failure.** If one specialist raises a `MedAgentError`, it is recorded as a failure and the report is built from the rest, with the failure noted. Only if **all** specialists fail does the run raise `AllAgentsFailedError`. Non-`MedAgentError` exceptions are bugs and stay loud.
- **Subgraphs.** The assessment graph can be embedded inside the follow-up graph.
- **`interrupt()`** for human approval ([section 12](#12-human-in-the-loop-approval)).
- **Checkpointer.** State is checkpointed in memory so a paused run can resume. It is in memory on purpose so patient data is never written to disk. A msgpack allowlist (`_CHECKPOINT_STATE_TYPES`) controls which types may be deserialised; a test fails if a new state type is missing from it.
- **Progress streaming.** Nodes emit `progress` events, which the WebSocket forwards as they happen.
- **Traced nodes.** Each node is wrapped by `traced_node`, so it appears as its own span.

## 8. Loop engineering

A ReAct loop can run away: too many turns, a hung provider, or a model stuck repeating itself. Every loop is bounded five ways:

| Bound | What it stops |
|---|---|
| **Turn cap** (`_REACT_MAX_ITERATIONS = 3` for the evidence agent) | endless think/act cycles |
| **Whole-loop deadline** (`AGENT_LOOP_TIMEOUT_SECONDS`, default 40 s) | a hung provider holding a request for minutes, since per-call timeouts add up across turns |
| **Prompt ceiling** (`AGENT_PROMPT_MAX_TOKENS`, default 4000) | context growing past what was intended |
| **Capped, untrusted tool output** (`AGENT_CONTEXT_FIELD_MAX_TOKENS`, default 1000) | one giant tool response flooding the prompt or injecting instructions |
| **Duplicate-call detection** | the model repeating the same search; a stalled turn forces a final answer |

Also:

- **Parallel tools.** When the model requests several tool calls in one turn, they run concurrently. Tools must therefore be safe to run concurrently.
- **Per-run token budget** (`AgentRunTracker`, `AGENT_MAX_TOKENS_PER_RUN`, default 8000). Each step records its token use; once the run is over budget, remaining steps are **skipped** rather than run.
- **Figure verification.** After an answer, `infra/verification.py` checks numbers in it (doses, percentages, lab values) against the tool results, the patient record and the question. Any figure that appears nowhere is flagged in the report with a warning. This is advisory (it never blocks an answer) and currently runs on the evidence agent's answers.

## 9. Harness engineering: errors, retries, circuit breakers, rate limits

"Harness" means everything around the model and the tools that decides how failures are handled. The goal is that a flaky dependency degrades the answer instead of hanging or crashing the request.

### 9.1 Classified errors

Anything we do not control (the LLM, an MCP server) raises an `UpstreamError` subclass (`ProviderError` or `MCPError`) carrying:

- `reason`: `timeout`, `connection_error`, `server_error`, `rate_limited`, `client_error`, or `circuit_open`,
- `retryable`: whether trying again can help,
- `retry_after`: the wait an upstream asked for, when known.

Each error also knows its HTTP status. `public_message()` returns fixed wording for clients, so raw upstream text (which can contain account ids or internal hostnames) is never returned.

### 9.2 One retry policy (`infra/retry.py`)

- Only **retryable** errors are retried. A bad request (4xx) or an unknown model is attempted once.
- Exponential backoff with jitter, 3 attempts.
- A rate limit is waited out **only** if the requested wait is at most `RETRY_MAX_WAIT_SECONDS` (default 10 s). Groq's daily cap says "try again in 8 minutes"; holding a doctor's request that long helps nobody, so it fails fast instead.
- Total time spent retrying is bounded by `RETRY_BUDGET_SECONDS` (default 45 s).
- SDK-level retries are turned off, because they stacked under ours (up to 9 HTTP calls per LLM call). Retries live in this one place.
- Each retry adds a span event and a metric.

### 9.3 Circuit breakers (`infra/circuit_breaker.py`)

One breaker per dependency: the LLM and each MCP server.

```
CLOSED ──5 consecutive upstream-health failures──► OPEN ──after 30 s──► HALF-OPEN
   ▲                                                  │  calls fail fast            │
   └──────────────── trial call succeeds ◄────────────┘◄──── trial call fails ──────┘
```

- **What counts as a failure:** timeouts, connection errors and 5xx only. A 4xx or a rate limit means the server is up, so it never trips the breaker.
- **OPEN:** calls fail immediately with the dependency's own error type and reason `circuit_open`, instead of waiting on a dead server.
- **HALF-OPEN:** after 30 s one trial call is allowed through; success closes the breaker.
- State is per process and is exported as a metric (`0/1/2`). The `/ready` endpoint reports it.

### 9.4 Rate limiting (`infra/rate_limiter.py`)

A token bucket per MCP server (`MCP_RATE_LIMIT_PER_SECOND`, default 5, burst `MCP_RATE_LIMIT_CAPACITY`, default 10). The free medical APIs have low ceilings, so calls are throttled before they ever reach retry.

### 9.5 Timeouts and honest HTTP codes

- Per-call timeouts: `LLM_TIMEOUT_SECONDS` (30 s), `MCP_TIMEOUT_SECONDS` (30 s).
- A rate limit surfaces as **429 with a `Retry-After` header** and honest wording ("the language model service is rate-limited; try again in about 40 seconds"), not a generic 502.

## 10. Context engineering

What goes into the model's prompt is bounded and labelled:

- **Budgets.** `truncate_text` cuts any single external field (a tool result, a literature dump) at `AGENT_CONTEXT_FIELD_MAX_TOKENS`, at a word boundary, and marks the cut. A final ceiling (`AGENT_PROMPT_MAX_TOKENS`) applies to the assembled prompt. Truncations are counted by source.
- **Token estimate.** 3.5 characters per token (`approx_token_count`). It is an estimate, chosen because a whitespace word count is fooled by long unspaced strings such as URLs.
- **Visit history.** Prior visits are replayed as short summaries (`AGENT_VISIT_SUMMARY_MAX_TOKENS`, default 250 each), because each stored report embeds the previous ones and replaying them verbatim would grow without bound.
- **Untrusted text is delimited.** `wrap_untrusted` wraps doctor, patient and external text in `<<< … >>>` with an instruction that it is data, never instructions (see [section 11](#11-guardrails-and-verification)).
- **Record vs evidence.** The prompt tells the model the patient's record and visit history describe *who the patient is* and are not evidence.
- **Patient-specific lookup.** Evidence retrieval is built from the patient's conditions and allergies, not just the message.

## 11. Guardrails and verification

Safety checks that must not depend on the model's mood live **outside** the LLM loop, in deterministic code:

| Guardrail | What it does |
|---|---|
| **Allergy conflict** | Checking a drug that matches a recorded allergy (case-insensitive substring) always appends a `⚠️ ALLERGY CONFLICT` line. It never relies on the LLM noticing. |
| **Output allergy cross-check** | If an LLM answer mentions a drug the patient is allergic to, a warning is added. |
| **Output sanity checks** | Flags suspiciously short answers and text that looks like a leaked system prompt. |
| **Prompt-injection delimiting** | `wrap_untrusted` (above). |
| **Figure verification** | Numbers not found in the sources are flagged. |
| **Lab flags** | Abnormal labs are flagged from each result's own reference range, outside any LLM. |
| **Changes since last visit** | Computed in code from the record and the last visit's snapshot, never by the model. |
| **Per-doctor isolation** | Enforced in the data layer, not in prompts. |

Every guardrail finding increments a metric by kind, so you can see how often each fires.

## 12. Human-in-the-loop approval

The WebSocket chat drafts a report but does **not** save it until the doctor decides. It is built on LangGraph's `interrupt()`.

```
Doctor sends a message
   → server streams {"type":"progress","step":...,"status":...}
   → graph reaches the approval node and calls interrupt(report)   ← paused
   → server sends {"type":"pending_approval","report":...}
Doctor replies:
   "approve"      → saved verbatim
   "reject"       → discarded
   any other text → treated as the doctor's edited version and saved instead
   → server resumes the graph and sends {"type":"saved"} or {"type":"rejected"}
```

- **A draft survives a dropped connection.** Reconnect with `?resume=1` to get it back (`"resumed": true`) and decide. Connecting without it discards any stale draft, so a new question is never mistaken for the reply to an old one.
- Drafts live only in server memory (they contain patient data), are private to the doctor who made them, and expire after `APPROVAL_DRAFT_TTL_MINUTES` (default 30). A server restart loses pending drafts.
- Every step (drafted, approved, edited, rejected, resumed, discarded) is written to the audit log.
- The plain REST `/assess` and `/followup` do **not** pause for approval; they auto-save. Approval exists only over the WebSocket.

## 13. RAG: guidelines search

RAG (retrieval-augmented generation) lets the agent answer from documents you supply.

**Ingestion** (`rag/pipeline.py`, run with `make ingest-guideline file=... source=...`):

```
PDF/text → extract text (pypdf) → chunk → embed (local model) → store in ChromaDB
```

- **Chunker** (`rag/chunker.py`): splits text into sections and then into windows of 512 words with 64 overlap. A line counts as a section header if it is a markdown `# heading`, a numbered heading such as `2.2 Risk prediction in people with CKD`, or an ALL-CAPS title of two or more real words. Table labels (`CKD G1`, `NA/NA`) and lone words do not qualify.
- **Embeddings** (`rag/embeddings.py`): `all-MiniLM-L6-v2` running locally on CPU. No patient text or guideline text goes to an external embedding service. Encodes are serialised for safe concurrent use.
- **Vector store** (`rag/vector_store.py`): a ChromaDB server in its own container. Each chunk keeps its `source` label and `section`.

**Retrieval** (`rag/retriever.py`, the `guideline_retriever` tool): embed the query, fetch the top-k nearest chunks, return them as evidence with source and title. The evidence agent calls it as `search_guidelines`.

Guidelines are not committed to the repository. The `guidelines_pdf/` folder is git-ignored. A short synthetic demo guideline lives in `docs/demo/`.

## 14. Memory and storage

### 14.1 The four kinds of memory, and where each lives

Agent memory is often described with four types borrowed from cognitive science:

| Type | Meaning | In MedAgent |
|---|---|---|
| **Working** | what the agent holds for the current task | LangGraph run state, the ReAct message buffer (`memory/session.py`) and the prompt. Lasts one request (approval drafts: up to 30 minutes). Bounded by the context budgets. |
| **Episodic** | specific past events | the `visits` table: each visit's question, report, date and a **snapshot of the record**. Recent visits are replayed as short summaries; the snapshot is the baseline for "what changed". The audit log also records events, for accountability only. |
| **Semantic** | facts | the patient record (conditions, allergies, medications, labs) in Postgres; guidelines in ChromaDB; FDA/PubMed/RxNorm/trial data from the MCP servers. |
| **Procedural** | how to do things | prompts, graph structure, triage rules, tools, guardrails. **Fixed in code on purpose**: an agent that rewrote its own clinical rules from feedback would change behaviour without review or tests. Doctor edits are logged (`report_edited`) for people to review. |

Principles the design follows:

- **Each memory type has its own trust level.** The patient record is authoritative and only changes through the API; guidelines and MCP data are evidence to cite; past reports are the agent's own earlier output and are labelled "not evidence" in prompts.
- **Model output never writes facts.** An LLM answer cannot add a medication or allergy to the record.
- **Only reviewed output becomes memory.** Over the WebSocket, only approved or edited reports are saved as visits; rejected drafts are never replayed.
- **Patient memory stays separate from domain knowledge** and scoped by `doctor_id`.

### 14.2 Storage

- **Postgres** holds patients, medications, lab results, visits, users, the audit log and an interactions log. The schema is owned by **Alembic** (`migrations/`, `make db-upgrade`). `doctor_id` is on every patient-data table.
- **Session buffer** (`memory/session.py`): a sliding window of conversation messages used by the ReAct agent.
- **Checkpoints** (LangGraph): in memory only, by design ([section 7](#7-graph-engineering-langgraph)).
- **ChromaDB**: guideline embeddings.

### 14.3 History is kept (append-only facts)

Saving a patient does not delete anything. `PatientStore.save_patient` reconciles the incoming record with what is stored, inside one transaction:

| Incoming change | What happens in the database |
|---|---|
| a new medication | a new active row (start date as given, or now) |
| a medication left out, or sent with `status: "stopped"` | the existing row is marked `stopped` with an `end_date`; the row stays |
| same medication, different dose, frequency or route | the old row is stopped, a new active row opens (start date now) |
| unchanged medication | nothing is written |
| a lab result | a new row; the same test at the same collection time is not duplicated (unique on patient, test, collected_at) |

Medications match by name, ignoring case and surrounding spaces. A re-save of an unchanged record writes nothing, which matters because `/assess`, `/followup` and the WebSocket re-save the loaded patient every time.

Reads still show the current view: `medications` = active only, `lab_results` = latest result per test. Conditions and allergies are stored on the patient row; their history is captured in the visit snapshots.

### 14.4 "What changed since the last visit"

1. When a visit is saved, `save_visit` stores `record_snapshot` on it: conditions, allergies, active medications (name, dose, frequency, route) and the latest lab per test.
2. When a patient is loaded, `get_patient` compares the current record with the newest visit's snapshot (`memory/record_changes.py`) and sets `changes_since_last_visit`.
3. The result lists: medications started, stopped and changed (before → after); conditions and allergies added or removed; labs with a newer result (previous → current value, unit, reference range, date).
4. The report shows it as a deterministic section, and the evidence agent gets the same list in its prompt (delimited as untrusted and length-capped).

Example section:

```
## Changes Since Last Visit (2026-09-27)

- Started: Lisinopril
- Stopped: Glimepiride
- Changed: Metformin: 500 mg -> 1000 mg
- New condition: hypertension
- HbA1c: 8.1 -> 9.8 % (reference 4.0-5.7), collected 2026-06-01
```

It states values only. There are no "significant change" thresholds and no interpretation; abnormal values are flagged separately by the Lab Flags section. With no earlier visit, or an earlier visit saved before snapshots existed, the section is left out. If nothing changed, it says "No recorded changes to medications, conditions, allergies or labs."

The schema change is migration `5b1e7c2d9a40` (adds `visits.record_snapshot` and the lab unique index, removing any duplicate lab rows first). It downgrades cleanly.

## 15. Security, privacy and audit

- **Authentication.** JWT (HS256) with password hashing. There is **no self-registration**: one admin is seeded from `ADMIN_BOOTSTRAP_USERNAME`/`PASSWORD` when the users table is empty, and every other account is created by an admin via `POST /admin/users`.
- **Per-doctor isolation.** `doctor_id` is stamped server-side from the creating doctor's identity and is immutable. Another doctor's patient returns **404, not 403**, so a patient's existence is never revealed. Admins bypass the filter.
- **Ownership on save.** Posting a patient ID that another doctor already owns is refused with **409** "This patient ID cannot be used." and changes nothing (`PatientOwnershipError`). Before this fix, such a request overwrote the other doctor's record. The 409 does reveal that the ID exists, which is hard to avoid while clients choose patient IDs.
- **Audit log.** Every patient-data access is recorded (created, viewed, assessed, follow-up, drug check, draft, approve, edit, reject) with the user and time, and is itself readable only for patients you own.
- **No patient data in telemetry.** Traces, logs and metric labels never carry a patient id, name, medication, allergy, or prompt/response text. Correlation uses `request_id`, `trace_id` and `run_id`. `tests/unit/infra/test_no_phi_in_telemetry.py` fails the build if a distinctive patient string appears in any span or log line.
- **Error hygiene.** Clients see fixed wording for upstream failures, not raw upstream text.
- **Local data.** Patient data stays in your Postgres and ChromaDB. The uvicorn access log is disabled (`--no-access-log`) because it prints raw URL paths.
- **LangSmith warning.** LangChain's tracing (`LANGCHAIN_TRACING_V2`) is **off by default** because it would send prompts, which contain patient data, to a third party. If turned on, the app logs a warning at startup.
- **Before real use:** change `JWT_SECRET_KEY` (the default is an insecure placeholder), set strong admin credentials, and put the API behind TLS.

## 16. Observability: logging, metrics, tracing, health

Three signals, tied together by ids.

### 16.1 Logging (structlog)

- JSON logs, ISO UTC timestamps, level, plus the fields each call site passes.
- Every line inside a request carries **`request_id`** (from the `X-Request-ID` header if it is well-formed, otherwise generated, and echoed in the response) and, when a span is active, **`trace_id`** and **`span_id`**.
- So one grep on a `request_id` gives every log line from that request across nodes, tools and LLM calls, and the `trace_id` opens the matching trace.
- Fields are chosen not to contain patient data (step, error type, reason, counts).

### 16.2 Metrics (Prometheus + Grafana)

`/metrics` is scraped by Prometheus; the provisioned **MedAgent Overview** dashboard (Grafana) shows:

- HTTP request rate and latency, by **route template** (`/patients/{patient_id}/assess`), not the raw path,
- LLM calls by outcome (success / rate_limited / timeout / server_error / client_error / circuit_open), latency, token spend,
- retries by reason,
- circuit-breaker state per dependency,
- MCP requests by outcome,
- specialist steps, and the evidence agent's ReAct-vs-fallback split,
- ReAct tool calls, guardrail findings, context truncations.

Rule: **metric labels come from small fixed sets** — never a patient id, query text, error message or a name the model chose. Unknown paths are recorded as `unmatched`, so scanners cannot create unbounded series.

### 16.3 Tracing (OpenTelemetry → Tempo)

One trace per request, as a tree:

```
POST /patients/{patient_id}/assess          (HTTP span)
 ├─ node.init_run
 ├─ node.triage
 ├─ node.drug_safety
 │    ├─ mcp_http.post./api/analysis/comprehensive   (MCP call)
 │    └─ llm.chat                                    (LLM call)
 ├─ node.evidence
 │    ├─ react.turn
 │    │    ├─ react.tool  (search_medical_literature)
 │    │    └─ react.tool  (search_guidelines)
 │    └─ llm.chat
 └─ node.report
```

- Spans: HTTP, graph nodes, ReAct turns, ReAct tools, LLM calls (using OpenTelemetry's GenAI attributes: model, input/output tokens, outcome; **never** prompt or completion text), MCP calls, WebSocket drafts.
- **Events** on spans record retries and circuit-breaker refusals, so a trace shows *why* something was slow.
- A W3C **`traceparent` header** is sent to the MCP servers so their side can join the trace.
- Spans are always created (they supply the `trace_id` for logs) but are **exported only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set**. The Docker stack sets it to Tempo. `OTEL_CONSOLE_EXPORTER=true` prints spans for local debugging, `OTEL_SAMPLE_RATIO` samples.
- To view: Grafana → Explore → Tempo → `{ resource.service.name = "medagent" }`, or the *Recent Traces* dashboard panel.

### 16.4 Health

- **`/health`** — liveness. "The process is up." Docker uses it; it does not depend on other services, so a down dependency never causes a restart loop.
- **`/ready`** — readiness (no auth). Checks Postgres (`SELECT 1`), ChromaDB (heartbeat), and each MCP server's health endpoint, and reports LLM and MCP breaker state. Returns `ok`, `degraded` (HTTP 200) or `down` (HTTP 503, only when Postgres is down). It shows only status words, never hostnames or error text, and it does **not** call the LLM (no token spend).

## 17. Testing strategy

| Kind | What | Where |
|---|---|---|
| **Unit** | ~438 tests. Mock LLM (`MockLLMProvider`), no network. Postgres-backed tests use a throwaway Docker container and skip if Docker is unavailable. | `tests/unit/` (mirrors `src/`) |
| **Fault injection** | Each failure mode (timeout, 5xx, 4xx, rate limit, breaker open) is simulated with a fake clock, so retries and breakers are tested without waiting. | `tests/unit/infra/test_fault_injection.py` |
| **Telemetry** | Span coverage, tracing setup, middleware labels, and the no-PHI check. | `tests/unit/infra/` |
| **Live evals** | Golden datasets against the **real** Groq model and real MCP servers, because mocks cannot catch real-model regressions. | `tests/eval/`, `make eval` |
| **Static** | ruff (lint + format), mypy, import-linter | `make pre-commit` |

Conventions: a new module gets a test file; anything that can only fail against the real model gets an eval case; a new shared resource needs a test that overlaps its use.

Commands: `make pre-commit` (format, lint, mypy, import check, unit tests) before finishing any task; `make ci-check` is the non-fixing version CI runs; `make eval` needs `GROQ_API_KEY` and `make docker-up-deps`.

## 18. Deployment and operations

`docker compose up -d --build` starts nine services:

| Service | Role |
|---|---|
| `medagent` | the API (runs `alembic upgrade head`, then uvicorn) |
| `postgres` | records, users, audit log |
| `chromadb` | guideline vectors |
| `medical-mcp-bridge`, `healthcare-mcp`, `research-mcp` | the three MCP servers |
| `prometheus` | scrapes `/metrics` |
| `tempo` | receives OTLP traces |
| `grafana` | dashboards and trace explorer |

Common commands:

```
make docker-up            # full stack
make docker-up-deps       # only backing services, for running the app on the host
make serve                # app on the host with reload
make db-upgrade           # migrations
make ingest-guideline file=guidelines_pdf/x.pdf source=my-source
```

Ports and URLs: API `:8000`, Grafana `:3000`, Prometheus `:9090`, Tempo `:3200`. If port 5432 is taken on your host, set `POSTGRES_HOST_PORT=5433`. When running tools on the host (not in Docker), use the host-mapped service ports rather than the Docker hostnames in `.env`.

CI runs `make ci-check` on every push. Live evals and Docker builds run on pull requests into `main`.

## 19. Configuration reference

All settings come from environment variables or `.env` through `Settings`. New settings must be added to `core/config.py` **and** `.env.example`. Never print `.env` values.

| Setting | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `groq` | which provider to use |
| `GROQ_API_KEY`, `GROQ_MODEL` | none, `openai/gpt-oss-20b` | model must support tool calling |
| `LLM_TIMEOUT_SECONDS`, `MCP_TIMEOUT_SECONDS` | 30, 30 | per-call timeouts |
| `RETRY_MAX_WAIT_SECONDS`, `RETRY_BUDGET_SECONDS` | 10, 45 | retry policy |
| `MCP_RATE_LIMIT_PER_SECOND`, `MCP_RATE_LIMIT_CAPACITY` | 5, 10 | outbound throttle per MCP server |
| `AGENT_MAX_TOKENS_PER_RUN` | 8000 | token budget per multi-agent run |
| `AGENT_CONTEXT_FIELD_MAX_TOKENS` | 1000 | cap per external text field |
| `AGENT_PROMPT_MAX_TOKENS` | 4000 | ceiling on the assembled prompt |
| `AGENT_VISIT_SUMMARY_MAX_TOKENS` | 250 | per stored-visit summary |
| `AGENT_LOOP_TIMEOUT_SECONDS` | 40 | whole ReAct loop deadline |
| `APPROVAL_DRAFT_TTL_MINUTES` | 30 | how long an unapproved draft is kept |
| `POSTGRES_DSN`, `CHROMA_HOST`, `CHROMA_PORT` | local defaults | storage |
| `MEDICAL_MCP_URL`, `HEALTHCARE_MCP_URL`, `RESEARCH_MCP_URL` | Docker service names | MCP endpoints |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | enables trace export |
| `OTEL_CONSOLE_EXPORTER`, `OTEL_SAMPLE_RATIO` | false, 1.0 | debug output, sampling |
| `LANGCHAIN_TRACING_V2` | false | LangSmith; sends patient data off-box, keep off |
| `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_EXPIRE_MINUTES` | placeholder, HS256, 60 | **change the secret** |
| `ADMIN_BOOTSTRAP_USERNAME`, `ADMIN_BOOTSTRAP_PASSWORD` | unset | seeds the first admin |
| `LOG_LEVEL` | INFO | logging level |

## 20. Known limitations

Stated plainly so they are not a surprise:

- **Answer groundedness is not fully guaranteed.** The prompt says to answer only from tool results and figures are checked, but a model can still state qualitative claims (for example which drug classes are safe with an allergy) that are not in the retrieved text. Anything clinical needs a doctor's review; this tool supports decisions, it does not make them.
- **Triage is keyword-based**, not semantic, so unusual phrasing falls back to the evidence agent.
- **`lab_interpreter` uses only each result's own reference range.** It does not adjust for age, sex or conditions.
- **Guideline coverage depends on what you ingest.** Retrieval quality depends on the chunker and the source text; a flowchart PDF gives few useful chunks.
- **Memory limits.** Past visits are replayed by recency, not relevance (the last five). Visits saved before record snapshots existed give no "what changed" baseline, so an existing patient's first follow-up after the upgrade has no Changes section. A patient row with an empty `doctor_id` (possible only in very old data) can no longer be saved until it is backfilled. Weight and height have no history.
- **Approval drafts are in memory.** A restart loses pending drafts, and the approval step exists only over the WebSocket.
- **Circuit-breaker state is per process**, not shared across replicas.
- **The Groq quota is a hard external limit.** When it is exhausted the API returns 429 with `Retry-After`, and reports degrade or fail accordingly.
- **Not built:** LLM prompt/response tracing (would need a self-hosted tool such as Langfuse or Phoenix because of PHI), log aggregation, alert rules, and database-query spans.

## 21. Glossary

| Term | Meaning |
|---|---|
| **MCP** | Model Context Protocol; here, three containers that expose medical data over HTTP |
| **ReAct** | Reason + Act loop: think, call a tool, observe, repeat |
| **LangGraph** | library used to build the workflows as graphs of nodes over shared state |
| **Node** | one step in a graph |
| **Subgraph** | a graph embedded as a node of another |
| **`interrupt()`** | LangGraph call that pauses a graph until it is resumed with a value |
| **Checkpointer** | stores graph state so a paused run can resume |
| **Specialist agent** | drug safety, evidence, or trial finder |
| **Circuit breaker** | stops calling a failing dependency for a while so callers fail fast |
| **Half-open** | breaker state that lets one trial call through to test recovery |
| **Retry-After** | header/hint saying how long to wait before retrying |
| **Token bucket** | rate-limit algorithm that refills tokens at a steady rate |
| **RAG** | retrieval-augmented generation: answer from retrieved documents |
| **Chunk** | a piece of a document stored and searched as one unit |
| **Embedding** | a vector representing the meaning of text |
| **Guardrail** | a deterministic safety check outside the model |
| **Untrusted text** | text from a doctor, patient or external source, delimited so the model treats it as data |
| **Span / Trace** | one timed operation / the tree of all operations for a request |
| **OTLP** | OpenTelemetry's export protocol |
| **Tempo** | the trace storage backend |
| **`request_id`, `trace_id`** | ids that tie logs and traces to one request |
| **Liveness / Readiness** | "is the process up" (`/health`) / "do its dependencies work" (`/ready`) |
| **PHI** | protected health information; kept out of all telemetry |
| **Working / episodic / semantic / procedural memory** | current-task state / past events (visits) / facts (record, guidelines, MCP data) / how-to (prompts, graph, rules; fixed in code) |
| **Append-only record** | medications and labs are never deleted; stops and dose changes close rows instead |
| **Record snapshot** | copy of the record stored on each visit; the baseline for the next comparison |
| **Changes Since Last Visit** | the code-computed difference between the record and the last snapshot |
| **Route template** | `/patients/{patient_id}/assess`, used in metrics and spans instead of the raw path |
| **Import check** | `make import-check`, which enforces the layer rules |
