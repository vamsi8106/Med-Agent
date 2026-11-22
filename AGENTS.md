# MedAgent

Clinical decision support agent for doctors. Checks drug interactions, retrieves treatment evidence, finds clinical trials, tracks patient history across sessions. All medical data from free open APIs via MCP servers. Patient data stays local (Postgres). Synthetic data only for demos.

**Stack:** Python 3.12+, uv, ruff, mypy, pytest, pydantic v2, structlog, httpx, FastAPI, LangGraph, asyncpg + Alembic. Observability: OpenTelemetry (Tempo), Prometheus, optional Langfuse prompt tracing.
**LLM:** `LLM_PROVIDER`: `groq` (cloud; demos with synthetic data), `openai_compatible` (a self-hosted model via an OpenAI-style API: Ollama, vLLM, LM Studio), `mock` (tests). Production refuses a third-party model unless `ALLOW_EXTERNAL_LLM_WITH_PHI=true`. No Groq imports outside `llm/groq_provider.py`.
**MCP:** medical-mcp (FDA/WHO/RxNorm/PubMed), healthcare-mcp (ICD-10/trials/calculator), med-research-mcp-suite (cross-DB analysis). All run locally, no API keys.
**Storage:** Postgres (patients, meds, labs, visits, users, audit log; schema owned by Alembic in `migrations/`), pgvector in the same Postgres (guideline embeddings + full-text index, versioned per document), sentence-transformers `all-MiniLM-L6-v2` (local, CPU-only torch).

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
│   ├── exceptions.py            # MedAgentError → UpstreamError (→ ProviderError, MCPError; carries `reason`, `retryable`, `retry_after`), ToolError, MemoryError, AuthError, PatientNotFoundError, PatientOwnershipError, RateLimitedError, AllAgentsFailedError (each carries its HTTP status_code)
│   ├── config.py                # Settings (pydantic-settings)
│   └── types.py                 # Enums: EvidenceGrade, InteractionSeverity, AgentRole
├── llm/                         # Phase 1 — provider layer
│   ├── registry.py              # ProviderRegistry: get_provider(name) → BaseLLMProvider
│   ├── chat_base.py             # ChatCompletionProvider: breaker, retry, llm.chat span, metrics shared by every provider; OpenAI-format message helpers
│   ├── groq_provider.py         # Groq implementation
│   ├── openai_compatible_provider.py  # self-hosted models over an OpenAI-style /chat/completions API (httpx)
│   └── mock_provider.py         # deterministic, for tests
├── infra/                       # Phase 1+ — cross-cutting
│   ├── logging.py               # structlog JSON setup
│   ├── retry.py                 # @retry: retries only `retryable` UpstreamErrors, honours Retry-After (fails fast past `max_wait`), bounded by a total `budget`
│   ├── rate_limiter.py          # token-bucket per API source (outbound, waits); KeyedRateLimiter per user/IP (inbound abuse limits, never waits, per process)
│   ├── circuit_breaker.py       # fail fast when a dependency (an MCP server, the LLM) is down; only timeouts/connection/5xx count, never 4xx or rate limits
│   ├── tracing.py               # OpenTelemetry setup (OTLP to Tempo iff OTEL_EXPORTER_OTLP_ENDPOINT), traced_node helper; optional Langfuse (LANGFUSE_ENABLED) on the same provider, llm_generation records prompts for Langfuse only
│   ├── metrics.py               # Prometheus catalogue: HTTP, LLM calls/latency/tokens, retries, breaker state, MCP, tool calls, ReAct paths, guardrail flags, truncations
│   ├── middleware.py            # request id, request logging, metrics + HTTP span (route templates only); SecurityMiddleware: body-size cap (413) + security headers
│   ├── context_budget.py        # token estimate (3.5 chars/token) + truncate_text for prompt inputs
│   ├── guardrails.py            # wrap_untrusted (removes instruction-like sentences, then delimits), allergy output check, output sanity checks, outbound_search_query (identifiers out of external API queries)
│   ├── verification.py          # deterministic check of an answer's numeric claims (doses, %, lab values) against its sources
│   └── agent_run.py             # AgentRunTracker: per-run token budget + step trace
├── tools/                       # Phase 2
│   ├── base.py                  # BaseTool ABC + ToolResult
│   ├── registry.py              # ToolRegistry: auto-discovery
│   ├── decorators.py            # @tool: auto JSON schema from type hints
│   ├── mcp/                     # HTTP clients: http_base.py (breaker/retry/rate-limit/tracing), medical.py, healthcare.py, research.py
│   └── custom/                  # interaction_checker, lab_interpreter, report_generator
├── agents/                      # Phase 3 (base + drug_safety), Phase 5 (rest)
│   ├── base.py                  # ReActAgent: LangGraph think → act loop. Bounded five ways: turn cap, whole-loop deadline, prompt ceiling, capped/untrusted tool output, duplicate-call detection (a stalled turn forces the answer). A turn's tool calls run concurrently, so tools must be concurrency-safe
│   ├── triage_agent.py          # routes to specialists
│   ├── drug_safety_agent.py     # interactions, adverse events
│   ├── evidence_agent.py        # ReAct over literature + guideline tools, fixed-pipeline fallback, figure verification on every answer
│   ├── trial_finder_agent.py    # ClinicalTrials.gov search
│   └── report_agent.py          # synthesizes multi-agent output
├── memory/                      # Phase 4
│   ├── patient_store.py         # Postgres CRUD for patient records (per-doctor isolation via doctor_id; saving another doctor's id → PatientOwnershipError 409). Medications/labs are append-only history: reconciled, never deleted; get_patient returns active meds + latest lab per test + changes_since_last_visit. log_interactions / get_interactions: insert-only interactions_log
│   ├── record_changes.py        # visit snapshot + deterministic diff (meds started/stopped/changed, conditions, allergies, newer labs) -- facts only, no clinical interpretation
│   ├── audit_log.py             # durable record of every patient-data access and every account change
│   ├── drafts.py                # Postgres approval drafts: TTL purge, per-patient delete (the saver itself is LangGraph's)
│   ├── rate_limits.py           # PostgresKeyedLimiter: abuse limits shared by every replica (rate_limit_buckets, row lock)
│   ├── retention.py             # opt-in purge of whole inactive patient records and old audit entries (RETENTION_DAYS / AUDIT_RETENTION_DAYS)
│   ├── session.py               # sliding-window conversation buffer (used by ReActAgent)
│   ├── schema.py                # table DDL + doctor_id statements
│   └── persistent.py            # asyncpg pool + connection handling
├── rag/                         # Phase 4
│   ├── embeddings.py            # sentence-transformers (local)
│   ├── vector_store.py          # pgvector store in Postgres: cosine search (HNSW), keyword search (full-text), versioned replace_document
│   ├── chunker.py               # section-header-aware; chunks sized to the embedding model's token limit, context prefix, reference lists dropped
│   ├── pipeline.py              # PDF → chunk → embed → store
│   ├── reranker.py              # CrossEncoderReranker: second retrieval stage (RAG_RERANKER_MODEL), re-scores the nearest chunks
│   └── retriever.py             # GuidelineRetrieverTool (BaseTool): RAG over ingested guidelines
├── auth/                        # JWT auth: security.py (hashing, tokens with a version claim), store.py (users: disable, password, revoke), dependencies.py (FastAPI deps; rejects disabled users and revoked tokens)
├── workflows/                   # Phase 5
│   ├── patient_assessment.py    # LangGraph: init_run → triage → parallel specialists → report. A specialist's MedAgentError becomes a recorded failure (degraded report), not an abort; all failing raises AllAgentsFailedError. Embeddable as a subgraph
│   ├── followup.py              # load patient → assessment subgraph → optional approval node (interrupt()); draft_followup / resolve_followup / pending_draft, progress streaming
│   ├── checkpointing.py         # in-memory (STATE_BACKEND=memory) or Postgres (postgres) checkpointer with a msgpack type allowlist; purge_expired_drafts
│   └── drug_check.py            # enumerate pairs → check → aggregate
├── cli.py                       # `medagent ingest-guideline` (RAG ingestion), `medagent purge-expired` (retention); separate process from the server
├── api/                         # HTTP + WebSocket layer
│   ├── state.py                 # AppState (all clients/agents, built once) + get_state dependency; build_specialists: the agents and their wiring, shared with the agent eval
│   ├── errors.py                # error → status/body mapping, doctor_scope, Retry-After
│   ├── health.py                # readiness aggregation for /ready (ok/degraded/down)
│   ├── schemas.py               # request/response bodies
│   └── routes/                  # system (health/ready/metrics), auth, patients (REST workflows), ws (approval chat)
└── app.py                       # Phase 6 — create_app(): lifespan, middleware, routers
```

Outside `src/`: `migrations/` (Alembic), `docker/` (MCP server images, with each server's audited `package.json` / `package-lock.json`), `ops/` (Prometheus/Grafana/Tempo/Caddy config), `docker-compose.prod.yml` (production overlay: Caddy TLS, no other public ports, required secrets, optional Ollama), `docs/deployment.md` (production runbook), `deploy/k8s/` (Kustomize: base, `overlays/local` for kind, `overlays/aws` for EKS + RDS + ALB, `components/aws` + `components/vllm`, `overlays/aws-generated` rendered by `scripts/render_aws_overlay.sh`), `deploy/eksctl/` (the cluster for AWS by hand, `docs/running.md` stage 4), `deploy/terraform/` (stage 5, AWS infrastructure: VPC, EKS, RDS, ECR, ACM, Secrets Manager, add-ons; `bootstrap/` state bucket; no secrets in state; runbook in its README), `tests/eval/` (live evals: agent eval + retrieval eval; how to run them on their own in `tests/eval/README.md`).

`tests/` mirrors `src/` 1:1. `docs/glossary.md` for canonical terms; `docs/info.md` explains every concept and flow; `docs/code-walkthrough.md` traces each request through the files and describes every module; `docs/running.md` has the commands to run it locally, in Docker, on kind and on AWS. Architecture decisions and their rationale live in commit messages and the README.

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

**Phase 2 — Tools & MCP:** `tools/` (base, registry, decorators, mcp clients, interaction_checker). Integration test with real medical-mcp. Gate: `search-drugs("metformin")` returns structured response via MCP.

**Phase 3 — Single Agent:** `agents/base.py` (ReAct loop) + `drug_safety_agent` + `memory/session.py`. Gate: "Check interactions for Metformin + Glimepiride" → correct response with citations.

**Phase 4 — Memory & RAG:** `memory/` (patient_store, persistent with Postgres) + `rag/` (full pipeline) + `guideline_retriever` + `lab_interpreter`. Gate: follow-up visit recalls patient, retrieves matching guidelines.

**Phase 5 — Multi-Agent:** remaining agents + `workflows/` + `report_generator`. Gate: complex patient → multi-agent → structured report.

**Phase 6 — Production:** `app.py` (FastAPI + WebSocket) + `infra/` (tracing, metrics, circuit_breaker, middleware) + Docker. Gate: `docker compose up` runs full system.

# Postgres Schema

`doctor_id` on every patient-data table below enforces per-doctor isolation
(each doctor sees only their own patients; admins bypass it) -- stamped
server-side from the creating doctor's JWT at patient creation, immutable
afterward. See `src/medagent/memory/schema.py`'s `ADD_DOCTOR_ID_STATEMENTS`.

Medications and lab results are append-only: a stopped medication keeps its
row (`status='stopped'`, `end_date`), a dose change closes the old row and
opens a new one, labs are unique per (patient, test, collected_at). Each visit
stores `record_snapshot`, the baseline for "Changes Since Last Visit".
`interactions_log` gets one row per drug pair each time an interaction check
runs (`/drug-check`, an assessment/follow-up where Drug Safety ran, an approved
or edited WebSocket report -- never a rejected draft); the API layer writes it
from the workflow outcome, `GET /patients/{id}/interactions` reads it.

```sql
patients    (id, name, age, sex, doctor_id, weight_kg, height_cm, conditions JSON, allergies JSON, created_at, updated_at)
medications (id, patient_id FK, doctor_id, name, brand_name, dose, frequency, route, start_date, end_date, status, created_at)
lab_results (id, patient_id FK, doctor_id, test_name, value, unit, reference_low, reference_high, is_abnormal, collected_at)
visits      (id, patient_id FK, doctor_id, visit_date, chief_complaint, assessment, plan, agent_session_id, record_snapshot JSONB, created_at)
interactions_log (id, patient_id FK, doctor_id, drug_a, drug_b, severity, description, source, checked_at)
guideline_documents (source PK, content_hash, embedding_model, chunker_version, chunk_count, ingested_at)
guideline_chunks    (id = sha256(source, position, text), source FK, position, section, text, embedding vector(384) HNSW, search tsvector GIN)
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
- Anything that can only fail against the real model or real MCP responses gets a case in `tests/eval/` -- unit tests use mocks and cannot see it. A new agent behaviour or workflow gets an `AgentCase` in `tests/eval/agent_cases/` (the eval runs assessment, follow-up and drug-check; new agents go through `build_specialists` so the eval runs them as the app does) (synthetic patient, source cited, `reviewed=False` until a clinician confirms it); metric definitions live in `tests/eval/agent_metrics.py` (unit-tested in `tests/unit/eval/`), gates and floors in `tests/eval/agent_thresholds.py`. Safety gates (allergy flag, severity floor, injection resisted, no identifier leak) must stay at 100 %; re-baseline floors only with the numbers in the commit message.
- A client for anything we don't control (LLM, MCP, a future API) must raise `UpstreamError` subclasses with a `reason` and `retryable`, own its retries in one place (turn any SDK's built-in retries off -- they stacked under ours, up to 9 HTTP calls per LLM call), put a circuit breaker in front, and emit metrics. A 4xx is never retryable and never trips the breaker; a rate limit is retried only if the requested wait is short. Add a fault-injection test for each new failure mode (`tests/unit/infra/test_fault_injection.py` shows the pattern with the `clock` fixture).
- Telemetry (traces, logs, metric labels) carries NO patient data: no patient id, name, medication, allergy, prompt or response text, no raw URL paths (use route templates). Correlate by `request_id` / `trace_id` / `run_id`. `tests/unit/infra/test_no_phi_in_telemetry.py` enforces it -- a new log field or span attribute must pass it. The one exception is opt-in prompt tracing: with `LANGFUSE_ENABLED`, `infra/tracing.py::llm_generation` puts messages and answers on spans of Langfuse's scope (`langfuse-sdk`), which only the Langfuse export receives (the OTLP/console exports drop that scope, tested in `test_tracing.py`). Never put prompt text on any other span. Production refuses a non-local `LANGFUSE_BASE_URL` unless `ALLOW_PROMPT_TRACING_WITH_PHI=true`. uvicorn runs with `--no-access-log` because its access log prints raw paths.
- API input is bounded: request-facing model fields carry max lengths / list sizes (`core/models.py`, `api/schemas.py`), bodies over `MAX_REQUEST_BODY_BYTES` get 413 (`SecurityMiddleware`). A new request field needs a bound.
- Abuse limits: login per username and per IP, agent runs per user (`KeyedRateLimiter`, `enforce_limit` in `api/state.py`), 429 + Retry-After. A new endpoint that spends LLM quota must use `agent_run_user` (or `enforce_limit`).
- `APP_ENV=production` refuses weak secrets at startup, hides the API docs and adds HSTS. The WebSocket authenticates with its first message (`{"type":"auth","token":...}`), never a URL token.
- Account changes (disable, password reset/change, revoke) bump the user's `token_version`; every token carries the version it was issued with, so old tokens stop working at once. All account changes are audited.
- Workflows return their structured results (`AssessmentOutcome`, `FollowupOutcome`, `DrugCheckOutcome`); persisting them (visits, `interactions_log`) is the API layer's job, so workflows never import `memory/`.
- State shared between replicas goes through `STATE_BACKEND`: `postgres` keeps approval drafts (LangGraph checkpointer) and abuse limits (`memory/rate_limits.py`) in the database. Anything else held in process memory must be per-pod by design (circuit breakers, outbound MCP buckets) or go through it. Migrations take a Postgres advisory lock (`migrations/env.py`), so concurrent `alembic upgrade head` is safe.
- The `checkpoint_*` schema is a frozen copy of the saver's migrations in `memory/schema.py`; after upgrading `langgraph-checkpoint-postgres`, `tests/unit/memory/test_drafts.py` (and the app at startup) tell you to add an Alembic migration.
- Guideline vectors are `vector(384)` (`EMBEDDING_DIMENSIONS`, all-MiniLM-L6-v2): another embedding size needs a migration. Bump `CHUNKER_VERSION` when chunking changes what a document becomes, so ingestion re-embeds instead of skipping it as unchanged.
- A change to chunking, the embedding model, re-ranking or retrieval needs `make rag-eval` before and after; report the numbers, including ms/query. Chunks must fit the embedding model's token limit (`EmbeddingModel.max_tokens`), or the rest of the chunk is silently not embedded.
- Retention deletes only when configured, whole records only (drafts included), and audits itself. Never add another code path that deletes patient data.
- `make deps-audit` (pip-audit) must stay clean; an ignored advisory needs a reason in the Makefile and is removed when a fix ships. `make mcp-audit` (npm audit over `docker/*/package-lock.json`) must stay clean of high/critical.
- Vendored MCP servers are pinned to commits in `scripts/setup_vendor_mcp.sh` and install our overlay lockfiles (`docker/<server>/`); an upgrade changes the commit, regenerates the overlay and re-runs `make mcp-audit` (`docs/deployment.md` §12).
- Any query sent to an external medical API goes through `infra/guardrails.py::outbound_search_query` (strips the patient's name and id, dates, numbers, contact details). Local searches (guideline pgvector) don't need it.
- `/health` is liveness (Docker uses it; must not flap on dependencies). `/ready` is readiness: Postgres down -> 503, other dependencies -> `degraded`; it never exposes hostnames or error text and never calls the LLM.
- Metric labels must come from a small fixed set -- never a patient id, query text, error message, or a name the model chose.
- Anything shared across requests (`AppState` clients, models) is used concurrently -- `HttpMCPClient` is re-entrant and `EmbeddingModel` serializes encodes for this reason. A new shared resource must be safe under overlapping use, with a test that overlaps it.
- A new type in graph state (`AssessmentState` / `FollowupState`, including nested models) must be added to `_CHECKPOINT_STATE_TYPES` in `workflows/checkpointing.py`; a test fails if one is missing. Graph nodes must not swallow non-`MedAgentError` exceptions -- those are bugs and stay loud.

# Commands

```
make install         # uv sync
make test            # uv run pytest
make unit-tests      # uv run pytest tests/unit/
make eval            # live evals: agent-eval gates + retrieval floors (real LLM + MCP servers; needs GROQ_API_KEY, `make docker-up-deps`)
make agent-eval      # agent eval report: every case end to end, rates (tool use, errors, hallucination proxies, safety, cost), eval-results/*.json + diff; args="--category safety --repeats 3 --judge"
make integration     # tests/integration: real embeddings + Postgres + medical-mcp bridge (skips what is unavailable)
make lint-check      # uv run ruff check .
make lint-fix        # uv run ruff check --fix .
make format-fix      # uv run ruff format .
make format-check    # uv run ruff format --check .
make type-check      # uv run mypy
make import-check    # import-linter: the layer rules above
make deps-audit      # pip-audit over the locked runtime dependencies (also in CI)
make mcp-audit       # npm audit over the MCP servers' committed lockfiles (also in CI)
make pre-commit      # format-fix + lint-fix + lint-check + type-check + import-check + unit-tests
make ci-check        # format-check + lint-check + type-check + import-check + unit-tests (what CI runs; no auto-fix)
make serve           # uv run uvicorn medagent.app:app --reload
make docker-up-deps  # backing services only (postgres with pgvector, 3 MCP servers) for bare-host dev
make docker-up       # full stack incl. the app; make docker-down / docker-logs
make db-upgrade      # alembic upgrade head (db-downgrade, db-revision name=... also exist)
make ingest-guideline file=... source=...   # RAG ingestion via the CLI
make rag-eval        # retrieval eval over guidelines_pdf/: recall@k, MRR, reaches-model, per chunking config
make purge-expired   # retention dry run (counts only); --execute via the CLI deletes
make k8s-local-up    # kind cluster, 3 replicas on deploy/k8s/overlays/local (make k8s-local-down)
make tf-check        # offline: terraform fmt + validate + Trivy scan of deploy/terraform (also in CI)
make tf-bootstrap / tf-init / tf-plan / tf-apply / tf-destroy   # AWS infrastructure (TF_ENV=demo|production, TF_STATE_BUCKET)
make tf-down         # full stage-5 teardown: app, wait for the ALB, terraform destroy, leftovers check
make aws-leftovers cluster=... [domain=...]   # lists anything still billing (exit 1); aws-wait-alb waits for the ALB to go
make aws-overlay version=...   # overlays/aws-generated from terraform output; aws-overlay-check renders it from fixtures (CI)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d   # production (see docs/deployment.md)
```

CI: `ci-check`, `mcp-audit` and `terraform` (`tf-check` + `aws-overlay-check`, offline) on every push; the live eval and Docker builds on PRs into `main` only.
