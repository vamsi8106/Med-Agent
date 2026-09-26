# MedAgent

Clinical decision support agent for doctors. Checks drug interactions, retrieves treatment evidence, finds clinical trials, tracks patient history across sessions. All medical data from free open APIs via MCP servers. Patient data stays local (Postgres + ChromaDB). Synthetic data only for demos.

**Stack:** Python 3.12+, uv, ruff, mypy, pytest, pydantic v2, structlog, httpx, FastAPI, LangGraph, asyncpg + Alembic.
**LLM:** Groq (initial) → swappable via `LLM_PROVIDER` env var. No Groq imports outside `llm/groq_provider.py`.
**MCP:** medical-mcp (FDA/WHO/RxNorm/PubMed), healthcare-mcp (ICD-10/trials/calculator), med-research-mcp-suite (cross-DB analysis). All run locally, no API keys.
**Storage:** Postgres (patients, meds, labs, visits, users, audit log; schema owned by Alembic in `migrations/`), ChromaDB (guideline embeddings, its own container), sentence-transformers `all-MiniLM-L6-v2` (local, CPU-only torch).

# User Flows

**New patient:** Doctor enters demographics + meds + labs → Triage routes to Drug Safety Agent (checks interactions via MCP) + Evidence Agent (searches PubMed + RAG guidelines) in parallel → Report Agent synthesizes → saves patient to Postgres.

**Follow-up:** Doctor references patient ID → agent loads full history from Postgres → checks what changed → runs relevant agents → updates records.

**Drug interaction:** Doctor asks "can I add Drug X?" → loads current meds from memory → runs all pairwise interaction checks via MCP → flags risks with severity + FDA data.

**Trial search:** Doctor asks about trials → Trial Finder searches ClinicalTrials.gov via MCP → filters by condition/phase/status → cross-references PubMed for results.

**Evidence lookup:** Doctor asks clinical question → the Evidence Agent runs a ReAct loop, choosing between guideline RAG and PubMed MCP searches → structures response with citations. If the model retrieves nothing or the loop fails, it falls back to the fixed retrieve-then-summarize pipeline. Every LLM answer passes the output guardrails (allergy cross-check, sanity validation) regardless of path.

# Structure

```
src/medagent/
├── core/                        # Phase 1 — shared kernel, zero external deps beyond pydantic
│   ├── interfaces.py            # ABCs: BaseLLMProvider, BaseTool, BaseAgent, BaseMemory
│   ├── models.py                # PatientContext, Medication, LabResult, Visit, DrugInteraction, Message, ToolCall, LLMResponse
│   ├── exceptions.py            # MedAgentError → UpstreamError (→ ProviderError, MCPError; carries `reason`, `retryable`, `retry_after`), ToolError, MemoryError, AuthError, PatientNotFoundError, AgentBudgetExceededError, AllAgentsFailedError (each carries its HTTP status_code)
│   ├── config.py                # Settings (pydantic-settings)
│   └── types.py                 # Enums: EvidenceGrade, InteractionSeverity, CKDStage, TrialPhase, AgentRole
├── llm/                         # Phase 1 — provider layer
│   ├── registry.py              # ProviderRegistry: get_provider(name) → BaseLLMProvider
│   ├── groq_provider.py         # Groq implementation
│   └── mock_provider.py         # deterministic, for tests
├── infra/                       # Phase 1+ — cross-cutting
│   ├── logging.py               # structlog JSON setup
│   ├── retry.py                 # @retry: retries only `retryable` UpstreamErrors, honours Retry-After (fails fast past `max_wait`), bounded by a total `budget`
│   ├── rate_limiter.py          # token-bucket per API source
│   ├── circuit_breaker.py       # fail fast when a dependency (an MCP server, the LLM) is down; only timeouts/connection/5xx count, never 4xx or rate limits
│   ├── tracing.py               # OpenTelemetry setup (OTLP to Tempo iff OTEL_EXPORTER_OTLP_ENDPOINT), traced_node helper
│   ├── metrics.py               # Prometheus catalogue: HTTP, LLM calls/latency/tokens, retries, breaker state, MCP, tool calls, ReAct paths, guardrail flags, truncations
│   ├── middleware.py            # request id, request logging, metrics + HTTP span (route templates only)
│   ├── context_budget.py        # token estimate (3.5 chars/token) + truncate_text for prompt inputs
│   ├── guardrails.py            # wrap_untrusted (prompt-injection delimiting), allergy output check, output sanity checks
│   ├── verification.py          # deterministic check of an answer's numeric claims (doses, %, lab values) against its sources
│   └── agent_run.py             # AgentRunTracker: per-run token budget + step trace
├── tools/                       # Phase 2
│   ├── base.py                  # BaseTool ABC + ToolResult
│   ├── registry.py              # ToolRegistry: auto-discovery
│   ├── decorators.py            # @tool: auto JSON schema from type hints
│   ├── mcp/                     # HTTP clients: http_base.py (breaker/retry/rate-limit/tracing), medical.py, healthcare.py, research.py
│   └── custom/                  # interaction_checker, patient_context, lab_interpreter, report_generator
├── agents/                      # Phase 3 (base + drug_safety), Phase 5 (rest)
│   ├── base.py                  # ReActAgent: LangGraph think → act loop. Bounded five ways: turn cap, whole-loop deadline, prompt ceiling, capped/untrusted tool output, duplicate-call detection (a stalled turn forces the answer). A turn's tool calls run concurrently, so tools must be concurrency-safe
│   ├── triage_agent.py          # routes to specialists
│   ├── drug_safety_agent.py     # interactions, adverse events
│   ├── evidence_agent.py        # ReAct over literature + guideline tools, fixed-pipeline fallback, figure verification on every answer
│   ├── trial_finder_agent.py    # ClinicalTrials.gov search
│   └── report_agent.py          # synthesizes multi-agent output
├── memory/                      # Phase 4
│   ├── patient_store.py         # Postgres CRUD for patient records (per-doctor isolation via doctor_id)
│   ├── audit_log.py             # durable record of every patient-data access
│   ├── session.py               # sliding-window conversation buffer (used by ReActAgent)
│   ├── schema.py                # table DDL + doctor_id statements
│   └── persistent.py            # asyncpg pool + connection handling
├── rag/                         # Phase 4
│   ├── embeddings.py            # sentence-transformers (local)
│   ├── vector_store.py          # ChromaDB adapter
│   ├── chunker.py               # section-header-aware, 512 tokens, 64 overlap
│   ├── pipeline.py              # PDF → chunk → embed → store
│   └── retriever.py             # GuidelineRetrieverTool (BaseTool): RAG over ingested guidelines
├── auth/                        # JWT auth: security.py (hashing, tokens), store.py (users), dependencies.py (FastAPI deps)
├── workflows/                   # Phase 5
│   ├── patient_assessment.py    # LangGraph: init_run → triage → parallel specialists → report. A specialist's MedAgentError becomes a recorded failure (degraded report), not an abort; all failing raises AllAgentsFailedError. Embeddable as a subgraph
│   ├── followup.py              # load patient → assessment subgraph → optional approval node (interrupt()); draft_followup / resolve_followup / pending_draft, progress streaming
│   ├── checkpointing.py         # in-memory checkpointer (PHI stays in process memory) with a msgpack type allowlist; purge_expired_drafts
│   └── drug_check.py            # enumerate pairs → check → aggregate
├── cli.py                       # `medagent ingest-guideline` (RAG ingestion), separate process from the server
├── health.py                    # readiness aggregation for /ready (ok/degraded/down)
├── api/                         # HTTP + WebSocket layer
│   ├── state.py                 # AppState (all clients/agents, built once) + get_state dependency
│   ├── errors.py                # error → status/body mapping, doctor_scope, Retry-After
│   ├── schemas.py               # request/response bodies
│   └── routes/                  # system (health/ready/metrics), auth, patients (REST workflows), ws (approval chat)
└── app.py                       # Phase 6 — create_app(): lifespan, middleware, routers
```

Outside `src/`: `migrations/` (Alembic), `docker/` (MCP server images), `ops/` (Prometheus/Grafana config), `tests/eval/` (live golden-dataset evals).

`tests/` mirrors `src/` 1:1. `docs/glossary.md` for canonical terms. Architecture decisions and their rationale live in commit messages and the README.

# Dependency Rules

```
core/     → nothing
infra/    → core/
llm/      → core/, infra/
tools/    → core/, infra/
memory/   → core/
rag/      → core/, infra/
auth/     → core/, memory/
agents/   → core/, infra/, tools/, memory/, rag/   (LLMs arrive via core.interfaces, not llm/)
workflows/→ core/, infra/, agents/
api/      → everything except app.py
app.py    → everything
```

Never import upward. `core/` never imports from `llm/`. `tools/` never imports from `agents/`. `make import-check` (import-linter, contracts in `pyproject.toml`) enforces this table on direct imports and runs in `pre-commit` and CI.

# Phases

**Phase 1 — Foundation:** `core/` (all files) + `infra/` (logging, retry, rate_limiter) + `llm/` (groq, mock, registry). Unit tests for models, config, retry, mock provider. Gate: `make pre-commit` green.

**Phase 2 — Tools & MCP:** `tools/` (base, registry, decorators, mcp clients, interaction_checker, patient_context). Integration test with real medical-mcp. Gate: `search-drugs("metformin")` returns structured response via MCP.

**Phase 3 — Single Agent:** `agents/base.py` (ReAct loop) + `drug_safety_agent` + `memory/session.py`. Gate: "Check interactions for Metformin + Glimepiride" → correct response with citations.

**Phase 4 — Memory & RAG:** `memory/` (patient_store, persistent with Postgres) + `rag/` (full pipeline) + `guideline_retriever` + `lab_interpreter`. Gate: follow-up visit recalls patient, retrieves matching guidelines.

**Phase 5 — Multi-Agent:** remaining agents + `workflows/` + `report_generator`. Gate: complex patient → multi-agent → structured report.

**Phase 6 — Production:** `app.py` (FastAPI + WebSocket) + `infra/` (tracing, metrics, circuit_breaker, middleware) + Docker. Gate: `docker compose up` runs full system.

# Postgres Schema

`doctor_id` on every patient-data table below enforces per-doctor isolation
(each doctor sees only their own patients; admins bypass it) -- stamped
server-side from the creating doctor's JWT at patient creation, immutable
afterward. See `src/medagent/memory/schema.py`'s `ADD_DOCTOR_ID_STATEMENTS`.

```sql
patients    (id, name, age, sex, doctor_id, weight_kg, height_cm, conditions JSON, allergies JSON, created_at, updated_at)
medications (id, patient_id FK, doctor_id, name, brand_name, dose, frequency, route, start_date, end_date, status, created_at)
lab_results (id, patient_id FK, doctor_id, test_name, value, unit, reference_low, reference_high, is_abnormal, collected_at)
visits      (id, patient_id FK, doctor_id, visit_date, chief_complaint, assessment, plan, agent_session_id, created_at)
interactions_log (id, patient_id FK, doctor_id, drug_a, drug_b, severity, description, source, checked_at)
```

# MCP Server Config

Each MCP server runs as its own container, reached over HTTP -- not spawned
as an in-process stdio subprocess -- so they scale, fail, and deploy
independently (see `docker-compose.yml`). `medical-mcp` is stdio-only (no
native HTTP mode), so it runs behind a small stdio<->HTTP bridge sidecar
(`docker/medical-mcp-bridge`); `healthcare-mcp` and `med-research-mcp-suite`
expose their own plain REST APIs directly (not real MCP-over-HTTP).

```yaml
medical-mcp-bridge: { url: http://medical-mcp-bridge:8080, contract: "POST /call-tool {name, arguments} -> {content}" }
healthcare-mcp:     { url: http://healthcare-mcp:3000, contract: "POST /mcp/call-tool {name, arguments}" }
research-mcp:       { url: http://research-mcp:3000, contract: "REST: /api/analysis/*, /api/trials/*, /api/fda/*" }
```

Tools available: `search-drugs`, `get-drug-details`, `search-drug-nomenclature`, `get-health-statistics`, `search-medical-literature`, `get-article-details`, `fda_drug_lookup`, `clinical_trials_search`, `lookup_icd_code` (ICD-10), `calculate_bmi`, `research_comprehensive_analysis`, `research_drug_safety_profile`.

# Custom Tools

| Tool | Does | Inputs → Outputs |
|---|---|---|
| `interaction_checker` | pairwise drug checks via MCP | `list[Medication]` → `list[DrugInteraction]` |
| `patient_context` | CRUD patient records | `patient_id` → `PatientContext` |
| `lab_interpreter` | flag abnormals against each result's own reference range (`PatientContext` is accepted but not yet used: no age/sex/condition adjustment) | `list[LabResult], PatientContext` → `list[LabFlag]` |
| `report_generator` | structure findings with citations | `AgentResult` → markdown report |
| `guideline_retriever` (in `rag/retriever.py`) | RAG over ingested guidelines | `query, top_k` → `list[ClinicalEvidence]` |

# Rules

- Type-annotate everything including `-> None`.
- Async for I/O, sync for CPU.
- `structlog` logger, never `print()`.
- Datetimes UTC, timezone-aware. No naive `datetime`.
- Raise `MedAgentError` subclasses, never bare `Exception`.
- Config via `Settings` singleton. New env var → `.env.example` + `config.py`.
- Synthetic patient data only. Names like "Patient Alpha", IDs like "P-TEST-001".
- No abstraction without a second caller.
- Test with `MockLLMProvider`. No network in unit tests (Postgres-backed tests use a throwaway Docker container and skip if Docker is unavailable).
- LLM prompts: wrap doctor/patient/external text with `wrap_untrusted`, and cap it with `truncate_text` (`AGENT_*_TOKENS` settings). Deterministic safety checks (allergies, interactions, lab flags) stay outside any LLM loop.
- Anything that can only fail against the real model or real MCP responses gets a case in `tests/eval/` -- unit tests use mocks and cannot see it.
- A client for anything we don't control (LLM, MCP, a future API) must raise `UpstreamError` subclasses with a `reason` and `retryable`, own its retries in one place (turn any SDK's built-in retries off -- they stacked under ours, up to 9 HTTP calls per LLM call), put a circuit breaker in front, and emit metrics. A 4xx is never retryable and never trips the breaker; a rate limit is retried only if the requested wait is short. Add a fault-injection test for each new failure mode (`tests/unit/infra/test_fault_injection.py` shows the pattern with the `clock` fixture).
- Telemetry (traces, logs, metric labels) carries NO patient data: no patient id, name, medication, allergy, prompt or response text, no raw URL paths (use route templates). Correlate by `request_id` / `trace_id` / `run_id`. `tests/unit/infra/test_no_phi_in_telemetry.py` enforces it -- a new log field or span attribute must pass it. uvicorn runs with `--no-access-log` because its access log prints raw paths.
- `/health` is liveness (Docker uses it; must not flap on dependencies). `/ready` is readiness: Postgres down -> 503, other dependencies -> `degraded`; it never exposes hostnames or error text and never calls the LLM.
- Metric labels must come from a small fixed set -- never a patient id, query text, error message, or a name the model chose.
- Anything shared across requests (`AppState` clients, models) is used concurrently -- `HttpMCPClient` is re-entrant and `EmbeddingModel` serializes encodes for this reason. A new shared resource must be safe under overlapping use, with a test that overlaps it.
- A new type in graph state (`AssessmentState` / `FollowupState`, including nested models) must be added to `_CHECKPOINT_STATE_TYPES` in `workflows/checkpointing.py`; a test fails if one is missing. Graph nodes must not swallow non-`MedAgentError` exceptions -- those are bugs and stay loud.

# Commands

```
make install         # uv sync
make test            # uv run pytest
make unit-tests      # uv run pytest tests/unit/
make eval            # live golden-dataset evals: real Groq + real MCP servers (needs GROQ_API_KEY, `make docker-up-deps`)
make lint-check      # uv run ruff check .
make lint-fix        # uv run ruff check --fix .
make format-fix      # uv run ruff format .
make format-check    # uv run ruff format --check .
make type-check      # uv run mypy
make import-check    # import-linter: the layer rules above
make pre-commit      # format-fix + lint-fix + lint-check + type-check + import-check + unit-tests
make ci-check        # format-check + lint-check + type-check + import-check + unit-tests (what CI runs; no auto-fix)
make serve           # uv run uvicorn medagent.app:app --reload
make docker-up-deps  # backing services only (postgres, chromadb, 3 MCP servers) for bare-host dev
make docker-up       # full stack incl. the app; make docker-down / docker-logs
make db-upgrade      # alembic upgrade head (db-downgrade, db-revision name=... also exist)
make ingest-guideline file=... source=...   # RAG ingestion via the CLI
```

CI: `ci-check` on every push; the live eval and Docker builds on PRs into `main` only.
