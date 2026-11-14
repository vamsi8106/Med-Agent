# MedAgent code walkthrough

This explains how the code runs and what every file in `src/medagent/` does. It covers where execution starts, what happens on each request, where results go, and the role of each module.

For the concepts behind the design (ReAct, circuit breakers, memory types, security), see [`info.md`](info.md). For running it in production, see [`deployment.md`](deployment.md).

## Contents

1. [Entry points](#1-entry-points)
2. [Layers and the import rule](#2-layers-and-the-import-rule)
3. [Startup](#3-startup)
4. [What happens on every request](#4-what-happens-on-every-request)
5. [Flow: full assessment (`POST /patients/{id}/assess`)](#5-flow-full-assessment)
6. [Flow: follow-up (`POST /patients/{id}/followup`)](#6-flow-follow-up)
7. [Flow: drug check (`POST /drug-check`)](#7-flow-drug-check)
8. [Flow: WebSocket approval chat (`/ws/{patient_id}`)](#8-flow-websocket-approval-chat)
9. [Flow: login and account management](#9-flow-login-and-account-management)
10. [Flow: CLI (guideline ingestion, retention)](#10-flow-cli)
11. [Where the output goes](#11-where-the-output-goes)
12. [File reference](#12-file-reference)
13. [Outside `src/`](#13-outside-src)
14. [Tests](#14-tests)

---

## 1. Entry points

MedAgent can start in three ways.

| Entry | Code | Started by |
|---|---|---|
| **The API server** (HTTP + WebSocket) | `src/medagent/app.py`. `create_app()` builds the FastAPI app, and the last line, `app = create_app()`, is what Uvicorn loads. | `make serve` (`uvicorn medagent.app:app --reload`), or in Docker `alembic upgrade head && uvicorn medagent.app:app ...` (`docker-compose.yml`) |
| **The CLI** | `src/medagent/cli.py`, `main()` | `python -m medagent.cli <command>`, or the `medagent` script (`pyproject.toml` `[project.scripts]`), or `make ingest-guideline` / `make purge-expired` |
| **Database migrations** | `migrations/env.py` + `migrations/versions/*.py` | `alembic upgrade head` (`make db-upgrade`), run automatically before the server in Docker |

---

## 2. Layers and the import rule

Each folder under `src/medagent/` is one layer. A layer may import only the layers listed for it, never one above it. `make import-check` (import-linter, contracts in `pyproject.toml`) fails the build otherwise.

```
core/      → nothing                 shared definitions
infra/     → core                    reliability, safety, observability
llm/       → core, infra             model providers
tools/     → core, infra             MCP clients, deterministic tools
rag/       → core, infra             guideline search
memory/    → core                    Postgres
auth/      → core, memory            users, tokens
agents/    → core, infra, tools, memory, rag     (the LLM arrives via core.interfaces)
workflows/ → core, infra, agents     LangGraph graphs (no database access)
api/       → everything except app.py
app.py     → everything
```

Two consequences show up throughout the code:

- **Agents never import `llm/`.** They receive a `BaseLLMProvider` (from `core/interfaces.py`) at construction, so tests can pass `MockLLMProvider` and production can pass Groq or a self-hosted model.
- **Workflows never write to the database.** They return an outcome object (`AssessmentOutcome`, `FollowupOutcome`, `DrugCheckOutcome`), and the API route saves it. All writes happen in one layer.

---

## 3. Startup

When Uvicorn imports `medagent.app`, `create_app()` runs:

1. **Settings.** `get_settings()` (`core/config.py`) reads the environment and `.env`. With `APP_ENV=production` it refuses to start on a weak `JWT_SECRET_KEY`, a short bootstrap password, or an external LLM without `ALLOW_EXTERNAL_LLM_WITH_PHI=true`.
2. **The FastAPI app** is created. In production `/docs`, `/redoc` and `/openapi.json` are switched off.
3. **Middleware** is added (`infra/middleware.py`):
   - `SecurityMiddleware` (inner): 413 for bodies over `MAX_REQUEST_BODY_BYTES`, security headers, HSTS in production;
   - `ObservabilityMiddleware` (outer): request id, request log, metrics, HTTP span.
4. **Error handler**: `api/errors.py::medagent_error_handler` turns any `MedAgentError` into its HTTP status and a fixed, safe message.
5. **Routers** from `api/routes/` are included: `system`, `auth`, `patients`, `ws`.

Then the **lifespan** runs when the server actually starts:

1. `configure_logging()` (`infra/logging.py`) and `configure_tracing()` (`infra/tracing.py`).
2. Warnings if LangSmith tracing is on or the placeholder JWT secret is used.
3. `AppState(settings)` (`api/state.py`) builds every long-lived object once:

   | Object | From | Used for |
   |---|---|---|
   | `PersistentStore` | `memory/persistent.py` | asyncpg pool (1–10 connections) |
   | `PatientStore`, `UserStore`, `AuditLogStore` | `memory/`, `auth/store.py` | database access |
   | `login_by_username`, `login_by_ip`, `agent_runs` | `infra/rate_limiter.py` (`KeyedRateLimiter`) | abuse limits |
   | `llm` | `llm/registry.py` | Groq, the self-hosted provider, or the mock |
   | `EmbeddingModel`, `VectorStore` | `rag/` | guideline search |
   | `MedicalMCPClient`, `HealthcareMCPClient`, `ResearchMCPClient` | `tools/mcp/` | medical data |
   | `TriageAgent`, `DrugSafetyAgent`, `EvidenceAgent`, `TrialFinderAgent`, `ReportAgent` | `agents/` | the specialists |
   | `checkpointer` + `followup_graph` | `workflows/checkpointing.py`, `workflows/followup.py` | the long-lived WebSocket graph that can pause for approval (built in `init()` for the Postgres checkpointer) |
   | abuse limiters | `infra/rate_limiter.py` or `memory/rate_limits.py`, by `STATE_BACKEND` | per process, or shared by every replica |

4. `AppState.init()` opens the connection pool and, if the users table is empty, creates the bootstrap admin from `ADMIN_BOOTSTRAP_USERNAME` / `ADMIN_BOOTSTRAP_PASSWORD`.
5. The state is stored on `app.state.medagent`; every route gets it through `get_state()`.

On shutdown the pool is closed and the tracer flushed.

---

## 4. What happens on every request

Before any endpoint code runs:

```
client
  │
  ▼
ObservabilityMiddleware      (infra/middleware.py)
  • request_id (from X-Request-ID or new), bound to every log line
  • HTTP span, labelled with the route template (never the raw path)
  • on finish: http_request log line + metrics (count, latency)
  ▼
SecurityMiddleware           (infra/middleware.py)
  • Content-Length or streamed body over the cap → 413
  • adds nosniff, X-Frame-Options DENY, no-referrer, Cache-Control no-store, HSTS (prod)
  ▼
route function               (api/routes/*.py)
  • Depends(get_current_user) → auth/dependencies.py:
      decode JWT (auth/security.py) → load user (auth/store.py)
      → reject unknown / disabled / token-version mismatch (401)
  • Depends(agent_run_user) on LLM-spending endpoints → enforce_limit (api/state.py) → 429
  • request body validated by pydantic (core/models.py, api/schemas.py) → 422 on bad input
```

If anything raises a `MedAgentError`, `api/errors.py` maps it, for example:
- `PatientNotFoundError` → 404;
- `PatientOwnershipError` → 409;
- `RateLimitedError` → 429 with `Retry-After`;
- `MCPError` → 502 "an external medical data source was unavailable".

---

## 5. Flow: full assessment

`POST /patients/P-TEST-001/assess` with `{"message": "Check interactions for current medications"}`.

**1. Route: `api/routes/patients.py::assess_patient`**
1. The user passes `agent_run_user`, so they're authenticated and within their agent-run limit.
2. `patient_store.get_patient(id, doctor_scope(user))` loads the record. A doctor sees only their own patients; an admin sees all. If nothing is found, the response is 404.
3. `memory/patient_store.py::_load` builds the `PatientContext`:
   - active medications;
   - the latest lab per test;
   - the last 5 visits;
   - `changes_since_last_visit`, computed by `memory/record_changes.py::compute_changes` against the newest visit's snapshot.

**2. Workflow: `workflows/patient_assessment.py::run_patient_assessment`** builds and runs the LangGraph:

```
START
  │
init_run          AgentRunTracker (infra/agent_run.py): token budget, run_id
  │
triage            agents/triage_agent.py: keywords → which specialists
  │
  ├──► drug_safety_node     (only if ≥2 drugs and routed)
  ├──► evidence_node        (default)
  └──► trial_finder_node    (if "trial"/"study" in message)
         │   run in parallel; each wrapped by _run_specialist:
         │   • skipped if the token budget is spent
         │   • MedAgentError → recorded AgentFailure (degraded report), not a crash
  │
report_node       all failed → AllAgentsFailedError (502, or 429 if all rate-limited)
                  else report_agent → markdown + completeness note
                  + collects the Drug Safety interactions
  │
END  →  AssessmentOutcome(report, interactions)
```

**3. Inside the specialists**

- **Drug Safety** (`agents/drug_safety_agent.py`):
  1. Takes the drug names from the record and the message.
  2. `tools/custom/interaction_checker.py` checks every pair through `tools/mcp/research.py` (`comprehensive_analysis`) and reads the severity from the response.
  3. Checks the drugs against recorded allergies, in code.
  4. Asks the LLM to summarise, with the external text wrapped by `wrap_untrusted` and capped by `truncate_text`.
  5. Validates the output, appends the citations and any `⚠️ ALLERGY CONFLICT` lines, and scans the LLM's own text for allergy mentions.

- **Evidence** (`agents/evidence_agent.py`):
  1. Builds a patient preamble: conditions, allergies, medications, recent visits, and changes since the last visit, all wrapped as untrusted text.
  2. Runs a ReAct loop (`agents/base.py::ReActAgent`, max 3 turns) over two tools: `search_medical_literature` (`tools/mcp/medical.py` → PubMed) and `search_guidelines` (`rag/retriever.py` → ChromaDB).
  3. If the model retrieves nothing or the loop fails, it falls back to the fixed retrieve-then-summarize pipeline.
  4. Every answer goes through the figure check (`infra/verification.py`) and the allergy cross-check (`infra/guardrails.py`).

- **Trial Finder** (`agents/trial_finder_agent.py`): takes the first recorded condition, calls `tools/mcp/healthcare.py::clinical_trials_search` (ClinicalTrials.gov), and asks the LLM to summarise.

- **Report** (`agents/report_agent.py` → `tools/custom/report_generator.py`) builds the markdown:
  - patient header, then one section per specialist;
  - **Known Allergies**;
  - **Lab Flags** (`tools/custom/lab_interpreter.py`, abnormal only);
  - **Changes Since Last Visit** (`format_changes`).

**4. Every outbound call** (LLM or MCP) goes through the same harness:
- `infra/retry.py` (retryable errors only, Retry-After honoured, total budget);
- `infra/circuit_breaker.py`;
- `infra/rate_limiter.py` (MCP);
- a tracing span and metrics.

For the LLM this lives in `llm/chat_base.py`; for MCP in `tools/mcp/http_base.py`.

**5. Back in the route**, in order:
1. `save_patient(patient)`: reconciles medications and labs; nothing is deleted.
2. `save_visit(...)`: a new visit with a `record_snapshot`, the baseline for next time.
3. `log_interactions(...)`: one `interactions_log` row per drug pair.
4. `audit_log.record(user, id, "assessment_run")`.
5. Response: `{"report": "<markdown>"}`.

---

## 6. Flow: follow-up

`POST /patients/{id}/followup` → `api/routes/patients.py::followup_visit` → `workflows/followup.py::run_followup`.

```
load_patient   memory.get_patient(id, doctor_scope)   (404 if not theirs)
  │
assess         the assessment graph from section 5, embedded as a subgraph
  │
END  →  FollowupOutcome(report, context, interactions)
```

The route then saves the patient, the visit and the interactions, and writes the audit entry `followup_run`, exactly like the assessment. The difference is that the workflow loads the patient itself (through `BaseMemory`), so the "what changed since last visit" section reflects the stored history.

---

## 7. Flow: drug check

`POST /drug-check` with `{"patient_id": "P-TEST-001", "new_drug": "Ibuprofen"}` → `api/routes/patients.py::drug_check` → `workflows/drug_check.py::run_drug_check`.

```
load_patient         current medications from Postgres
  │
check_interactions   message = "Check interactions for <current meds>, <new drug>"
                     drug_safety.run_result(...)   (section 5, Drug Safety)
  │
END  →  DrugCheckOutcome(answer, interactions)
```

The route records the pairs in `interactions_log` and the audit entry `drug_check_run`, then returns `{"answer": ...}`. No visit is saved.

---

## 8. Flow: WebSocket approval chat

`/ws/P-TEST-001` (`api/routes/ws.py`). This is the only path where a human approves before anything is saved.

1. **Connect, then authenticate with the first message.**
   - The client must send `{"type":"auth","token":"..."}` within 10 s. The token is never put in the URL.
   - `auth/dependencies.py::user_for_ws_token` checks it; on failure the socket closes with code 1008.
2. **Resume or start fresh.**
   - With `?resume=1`, a pending draft for this user and patient is sent back (`"resumed": true`).
   - Without it, any old draft is discarded, so a new question is never mistaken for a reply to an old draft.
3. **The doctor sends a question.**
   - It's limited to 4000 characters and counts against the agent-run limit.
   - `workflows/followup.py::draft_followup` runs the follow-up graph *with* the approval node, streaming `{"type":"progress",...}` frames as each step finishes.
4. **Pause.**
   - The approval node calls LangGraph `interrupt()`. The graph state (the draft) is held by the checkpointer (`workflows/checkpointing.py`): in process memory with `STATE_BACKEND=memory`, or in Postgres with `postgres`, so a resume can land on any replica.
   - The server sends `{"type":"pending_approval","report":...}` and audits `report_drafted`.
5. **The doctor decides.**
   - `resolve_followup` resumes the graph with the reply, and `interpret_decision` maps it: `approve` → save the draft; `reject` → save nothing; any other text → save that text as the edited report.
   - On approve or edit, `_apply_outcome` saves the patient, the visit and the interactions, and audits `report_approved` or `report_edited`.
   - On reject it audits `report_rejected` and saves nothing.
6. **Expiry.** Drafts older than `APPROVAL_DRAFT_TTL_MINUTES` are purged on each connection (`AppState.purge_expired_drafts`: the checkpointer scan in memory, one SQL delete in `memory/drafts.py` for Postgres). With `STATE_BACKEND=memory` a restart loses them.

---

## 9. Flow: login and account management

**Login: `POST /auth/token`** (`api/routes/auth.py`)
1. Rate limited per client IP and per username, giving 429 with `Retry-After`.
2. `auth/store.py::get_by_username` loads the user. If the user is unknown, a dummy hash is checked anyway, so the response time doesn't reveal whether the account exists.
3. `auth/security.py::verify_password` checks the password. A disabled user gets the same 401 as a wrong password.
4. `create_access_token(..., version=user.token_version)` issues a JWT whose `ver` claim is the user's current token version.

**Every later request.** `auth/dependencies.py::_resolve_user` re-reads the user and rejects the token if the user is disabled or `ver` no longer matches. This is how revoking works without storing tokens.

**Admin endpoints** (`/admin/users...`): create, list, disable/enable, reset password, revoke tokens. Users can change their own password with `/auth/change-password`. Every change except enable increases `token_version`, so existing tokens stop working immediately. Every change is written to the audit log.

---

## 10. Flow: CLI

`cli.py::main` parses the command and runs it in its own process (not inside the server).

| Command | Flow | Output |
|---|---|---|
| `ingest-guideline --file X --source Y` | read the PDF (`pypdf`) or text → `rag/pipeline.py`: `rag/chunker.py` (section headers, chunks sized to the model's token limit, context prefix, reference lists dropped) → `rag/embeddings.py` → `rag/vector_store.py` (ChromaDB) | chunk count on the terminal; chunks in ChromaDB |
| `purge-expired [--execute]` | `memory/retention.py::purge_expired`: does nothing unless `RETENTION_DAYS` / `AUDIT_RETENTION_DAYS` are set (minimum 365). Dry run by default. `--execute` deletes whole inactive records and old audit entries, and audits the purge. | counts only on the terminal, never names or IDs |

---

## 11. Where the output goes

| Output | Written by | Destination |
|---|---|---|
| Report / answer (JSON) | the route, from the workflow outcome | the doctor's client |
| WebSocket frames (`progress`, `pending_approval`, `saved`, `rejected`, `error`) | `api/routes/ws.py` | the doctor's client |
| Patients, medications, labs, visits | `memory/patient_store.py` | Postgres |
| Interaction checks | `memory/patient_store.py::log_interactions` | Postgres `interactions_log` |
| Audit entries | `memory/audit_log.py` | Postgres `audit_log` |
| Users | `auth/store.py` | Postgres `users` |
| Guideline chunks | `rag/pipeline.py` (CLI) | ChromaDB |
| Logs | `infra/logging.py` (structlog JSON) | stdout → `docker compose logs` |
| Traces | `infra/tracing.py` (OTLP) | Tempo → Grafana |
| Metrics | `infra/metrics.py` (`/metrics`) | Prometheus → Grafana |

No log line, span attribute or metric label carries patient data; `tests/unit/infra/test_no_phi_in_telemetry.py` enforces this.

---

## 12. File reference

### Top level

| File | Does |
|---|---|
| `app.py` | Builds the FastAPI app: lifespan (logging, tracing, `AppState`), middleware, error handler, routers; hides the API docs in production. |
| `cli.py` | Command line: `ingest-guideline` and `purge-expired`. |
| `__init__.py` | Marks the package (empty). |

### `core/`: shared definitions (imports nothing from the project)

| File | Does |
|---|---|
| `config.py` | `Settings` (pydantic-settings): every environment variable with its default and bounds. The production check (`_refuse_insecure_production_config`) and `is_local_host` (is an LLM endpoint on a private host?). |
| `models.py` | The data shapes used everywhere: `PatientContext`, `Medication`, `LabResult`, `Visit`, `RecordChanges`, `DrugInteraction`, `ClinicalEvidence`, `AgentResult`, `AgentFailure`, `Message`, `ToolCall`, `LLMResponse`, `User`. Request-facing fields carry size limits. |
| `exceptions.py` | The error tree. `MedAgentError` is the root; `UpstreamError` (with `reason`, `retryable`, `retry_after`) covers `ProviderError` and `MCPError`. Also `ToolError`, `MemoryError`, `AuthError`, `PatientNotFoundError`, `PatientOwnershipError`, `RateLimitedError`, `AllAgentsFailedError`, each with its HTTP status. |
| `interfaces.py` | Abstract base classes: `BaseLLMProvider`, `BaseTool`, `BaseAgent`, `BaseMemory`. |
| `types.py` | Enums: `EvidenceGrade`, `InteractionSeverity`, `AgentRole`. |

### `infra/`: reliability, safety, observability

| File | Does |
|---|---|
| `logging.py` | structlog JSON setup; request_id and trace_id bound to every line. |
| `tracing.py` | OpenTelemetry setup (OTLP to Tempo if `OTEL_EXPORTER_OTLP_ENDPOINT` is set, console exporter optional), `get_tracer`, and `traced_node`, which wraps a graph node in a span. |
| `metrics.py` | The Prometheus metrics: HTTP, LLM calls/latency/tokens, retries, breaker state, MCP requests, tool calls, ReAct vs. fallback, guardrail flags, truncations, abuse-limit hits. Labels come from small fixed sets. |
| `middleware.py` | `ObservabilityMiddleware` (request id, request log, metrics, HTTP span with the route template) and `SecurityMiddleware` (body-size cap → 413, security headers, HSTS). |
| `retry.py` | `@retry`: retries only retryable `UpstreamError`s with backoff, honours Retry-After (fails fast past `max_wait`), and stays within a total time budget. |
| `circuit_breaker.py` | Opens after repeated real outages (timeouts, connection errors, 5xx), fails fast while open, then tries again. 4xx and rate limits never count. |
| `rate_limiter.py` | `TokenBucket` for outbound calls (waits); `KeyedRateLimiter` for incoming abuse limits per user/IP, per process (never waits; returns the retry time; bounded memory). |
| `context_budget.py` | Token estimate (3.5 characters per token) and `truncate_text`, which records a truncation metric. |
| `guardrails.py` | `wrap_untrusted` (delimits outside text so it isn't followed as instructions), the allergy-mention check on output, and `validate_output` sanity checks. |
| `verification.py` | Finds doses, percentages and lab values in an answer that appear in none of its sources, and adds an `UNVERIFIED FIGURES` warning. |
| `agent_run.py` | `AgentRunTracker`: token budget per run, step records (completed / skipped / failed), and a summary for the logs. |

### `llm/`: model providers

| File | Does |
|---|---|
| `registry.py` | `ProviderRegistry.get_provider(name, settings)`: `groq`, `openai_compatible` (needs `LLM_BASE_URL` + `LLM_MODEL`) or `mock`. |
| `chat_base.py` | `ChatCompletionProvider`: what every real provider shares (breaker, retry, `llm.chat` span, metrics, `LLMResponse` assembly). Helpers to convert messages to the OpenAI format, parse tool calls, read Retry-After, and classify HTTP status codes. |
| `groq_provider.py` | Groq SDK calls (SDK retries off) and mapping of Groq errors, including its "try again in 8m" text. The only file that imports Groq. |
| `openai_compatible_provider.py` | A self-hosted model (Ollama, vLLM, LM Studio) via httpx `POST /chat/completions`, with the same error mapping. |
| `mock_provider.py` | `MockLLMProvider`: a fixed answer, for tests. |

### `tools/`: what agents call

| File | Does |
|---|---|
| `base.py` | `BaseTool` and `ToolResult` (success, data, error, error_type). |
| `decorators.py` | `@tool`: turns an async function into a tool, building its JSON schema from the type hints. |
| `registry.py` | `ToolRegistry`: the set of tools offered to the ReAct loop, with schema listing and dispatch. |
| `mcp/http_base.py` | `HttpMCPClient`: the shared HTTP client for all MCP servers. Breaker, retry, rate limit, tracing, metrics, error mapping to `MCPError`, `probe()` for `/ready`. Safe to use from several requests at once. |
| `mcp/medical.py` | `MedicalMCPClient` (medical-mcp, through the stdio→HTTP bridge): `search_drugs`, `search_medical_literature`. |
| `mcp/healthcare.py` | `HealthcareMCPClient` (healthcare-mcp): `clinical_trials_search`. |
| `mcp/research.py` | `ResearchMCPClient` (med-research-mcp-suite): `comprehensive_analysis` (drug pairs), `drug_safety_profile`. |
| `custom/interaction_checker.py` | `InteractionCheckerTool`: every drug pair → research-mcp → `DrugInteraction` (severity read from the structured `riskProfile`, description trimmed to the clinically relevant fields). |
| `custom/lab_interpreter.py` | `LabInterpreterTool`: flags labs outside their own reference range. |
| `custom/report_generator.py` | `ReportGeneratorTool`: the markdown report (header, specialist sections, Known Allergies, Lab Flags, Changes Since Last Visit) and `format_changes`. |

### `rag/`: guideline search

| File | Does |
|---|---|
| `chunker.py` | Splits text on section headers (markdown, numbered `3.1.2 Title`, all-caps multi-word lines), then into windows that fit the embedding model (token counts from its tokenizer, 32-token overlap). Each chunk has a `<source> › <section>` context prefix for embedding; `is_reference_list` drops bibliography chunks. |
| `embeddings.py` | `EmbeddingModel`: sentence-transformers `all-MiniLM-L6-v2` on CPU; one encode at a time so it can be shared safely. `max_tokens` (the model's limit) and `count_tokens` size the chunks. |
| `vector_store.py` | `VectorStore`: ChromaDB client (add, query, `delete_source`); accepts an injected client for tests and the retrieval eval. |
| `pipeline.py` | `IngestionPipeline`: text → chunks → embeddings (with context) → ChromaDB; re-ingesting a source replaces its chunks. |
| `retriever.py` | `GuidelineRetrieverTool`: query → top-k `ClinicalEvidence` passages. |

### `memory/`: Postgres

| File | Does |
|---|---|
| `persistent.py` | `PersistentStore`: asyncpg pool and `connect()`. |
| `schema.py` | All table SQL (create, doctor_id, audit log, record history, user management), used by both the migrations and the tests. |
| `patient_store.py` | `PatientStore`: `get_patient` (doctor-scoped current view + changes since the last visit); `save_patient` (row lock, 409 on another doctor's ID, medications reconciled, labs appended); `save_visit` (with snapshot); `log_interactions` / `get_interactions`. |
| `record_changes.py` | `record_snapshot` and `compute_changes`: the facts-only "what changed" comparison (medications started/stopped/changed, new conditions or allergies, newer labs). |
| `audit_log.py` | `AuditLogStore`: `record()` and `for_patient()` (newest first, paged). |
| `drafts.py` | Postgres approval drafts: `purge_expired_drafts` (one SQL delete past the TTL) and `delete_drafts_for_patients` (used by retention). |
| `rate_limits.py` | `PostgresKeyedLimiter`: the abuse-limit token bucket in `rate_limit_buckets` under a row lock, shared by every replica. |
| `retention.py` | `purge_expired`: opt-in deletion of whole inactive records and old audit entries; dry run by default; audits itself. |
| `session.py` | `SessionMemory`: sliding-window message buffer for one ReAct run. |

### `auth/`

| File | Does |
|---|---|
| `security.py` | `hash_password` / `verify_password`; `create_access_token` (with the `ver` claim) and `decode_token_claims`. |
| `store.py` | `UserStore`: create, get, count, list, `set_disabled`, `set_password`, `revoke_tokens` (the last three increase `token_version`). |
| `dependencies.py` | `get_current_user` and `require_admin` for HTTP; `user_for_ws_token` for the WebSocket. Reject unknown users, disabled users and revoked tokens. |

### `agents/`

| File | Does |
|---|---|
| `base.py` | `ReActAgent`: a LangGraph think → act loop with 5 limits (turn cap, deadline for the whole loop, prompt ceiling, capped untrusted tool output, duplicate-call detection). A turn's tool calls run concurrently. |
| `triage_agent.py` | `TriageAgent`: keyword routing to `drug_safety` / `evidence` / `trial_finder` (defaults to evidence). |
| `drug_safety_agent.py` | `DrugSafetyAgent`: pair checks, allergy conflicts in code, LLM summary with citations; `run_result()` returns the structured interactions as well. |
| `evidence_agent.py` | `EvidenceAgent`: patient preamble, ReAct over literature and guidelines, fallback pipeline, figure check, allergy cross-check. |
| `trial_finder_agent.py` | `TrialFinderAgent`: ClinicalTrials.gov search by the patient's condition, then summary. |
| `report_agent.py` | `ReportAgent`: hands the specialist results to the report generator. |

### `workflows/`: LangGraph graphs (no database writes)

| File | Does |
|---|---|
| `patient_assessment.py` | The core graph: `init_run → triage → parallel specialists → report`. Failure and budget handling; returns `AssessmentOutcome(report, interactions)`; `build_assessment_graph` is reused as a subgraph. |
| `followup.py` | `load_patient → assessment subgraph → (approval node with interrupt())`. `run_followup` for REST; `draft_followup`, `pending_draft`, `resolve_followup` and `discard_draft` for the WebSocket; progress events. |
| `drug_check.py` | `load_patient → check_interactions`; returns `DrugCheckOutcome(answer, interactions)`. |
| `checkpointing.py` | The checkpointer: in memory (`build_checkpointer`) or Postgres (`open_postgres_checkpointer`, which refuses an out-of-date schema), with an allowlist of types it may save (`_CHECKPOINT_STATE_TYPES`), and `purge_expired_drafts` for the in-memory one. |

### `api/`: HTTP and WebSocket layer

| File | Does |
|---|---|
| `state.py` | `AppState` (section 3), `get_state` (works for HTTP and WebSocket), `enforce_limit` (429 + metric). |
| `errors.py` | `medagent_error_handler`, `error_payload`, `retry_after`, `doctor_scope` (doctor → own id, admin → no filter), `require_doctor_id`. |
| `schemas.py` | Request and response bodies: login token, user creation, passwords, assess and drug-check bodies, with size limits; `MAX_MESSAGE_CHARS`. |
| `health.py` | `check_readiness`: probes Postgres, ChromaDB and each MCP server (with a timeout) and reads the breaker states; ok / degraded / down; never hostnames or error text. |
| `routes/system.py` | `GET /health` (liveness), `GET /ready` (readiness; 503 when Postgres is down), `GET /metrics`. |
| `routes/auth.py` | `POST /auth/token`, `POST /auth/change-password`, `POST /admin/users`, `GET /admin/users`, `/admin/users/{username}/disable`, `/enable`, `/reset-password`, `/revoke-tokens`. |
| `routes/patients.py` | `POST /patients`, `GET /patients/{id}`, `GET /patients/{id}/audit-log`, `GET /patients/{id}/interactions`, `POST /patients/{id}/assess`, `POST /patients/{id}/followup`, `POST /drug-check`. |
| `routes/ws.py` | `/ws/{patient_id}`: first-message auth, resume, drafting with progress, approve/edit/reject, saving on approval. |

---

## 13. Outside `src/`

| Path | Does |
|---|---|
| `migrations/` | Alembic: `env.py` (connection), `versions/` (initial schema → audit log → doctor_id → record history → user management). |
| `docker/` | Images for the three MCP servers, plus the small stdio→HTTP bridge for medical-mcp (`docker/medical-mcp-bridge/server.js`). |
| `ops/` | Prometheus scrape config, Grafana data sources and dashboard, Tempo config, Caddyfile (production TLS). |
| `docker-compose.yml` | Development stack: 9 services, every port bound to 127.0.0.1. |
| `docker-compose.prod.yml` | Production overlay: Caddy (HTTPS) as the only public service, required secrets, optional Ollama. |
| `deploy/k8s/` | Kustomize manifests for several replicas: `base/`, `overlays/local/` (kind), `overlays/aws/` (EKS + RDS + ALB). |
| `Dockerfile` | The app image (CPU-only torch, non-root user). |
| `scripts/setup_vendor_mcp.sh` | Clones and builds the two MCP servers that come from GitHub into `vendor/`. |
| `Makefile` | Every developer command (`make pre-commit`, `make eval`, `make serve`...). |
| `pyproject.toml`, `uv.lock` | Dependencies, tool configuration, import-linter contracts. |

---

## 14. Tests

`tests/` follows the same layout as `src/`.

| Folder | What it tests | Needs |
|---|---|---|
| `tests/unit/<layer>/` | Each module, with `MockLLMProvider` and fakes | Nothing; Postgres-backed tests start a throwaway container and skip without Docker |
| `tests/unit/test_app.py` | The whole API through FastAPI's `TestClient`: auth, limits, isolation, WebSocket, interaction log | Docker (Postgres) |
| `tests/unit/infra/test_fault_injection.py` | Each failure mode (timeout, 5xx, 4xx, 429, breaker open) with a fake clock | Nothing |
| `tests/unit/infra/test_no_phi_in_telemetry.py` | No patient data in any span or log line | Nothing |
| `tests/integration/` | Real embeddings, a real Postgres container and the real medical-mcp bridge (`make integration`) | Docker, `make docker-up-deps` |
| `tests/eval/` | Golden cases against the real model and real MCP servers (`make eval`), and the retrieval eval (`make rag-eval`: recall@k, MRR, reaches-model) | `GROQ_API_KEY` and the MCP containers; the guideline PDFs for the retrieval eval |
