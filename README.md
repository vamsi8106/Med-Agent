# MedAgent

Clinical decision-support agent for doctors: checks drug interactions (including against recorded allergies), retrieves treatment evidence tailored to the patient's conditions, finds clinical trials, and flags abnormal labs — with each doctor scoped to only their own patients. Built on LangGraph, FastAPI and Postgres (with pgvector for guideline search), with all medical data sourced from free MCP servers (FDA/WHO/RxNorm/PubMed, ICD-10/trials/calculators, cross-database analysis).

Two ways to run it locally:

- **[Option A — Hybrid local dev](#option-a-hybrid-local-dev)**: the app runs bare on your host (fast reload via `uvicorn --reload`), backing services (Postgres with pgvector, the 3 MCP servers) run in Docker.
- **[Option B — Full Docker stack](#option-b-full-docker-stack)**: everything, including the app itself, runs in Docker — closest to the production topology.

Both are exercised with the same `curl` walkthrough below. **[docs/running.md](docs/running.md)** has every stage's commands in one place: local, Docker, Kubernetes on kind, and AWS.

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose
- Node.js/npm (only needed once, to vendor the two MCP servers built from source)
- A [Groq API key](https://console.groq.com/keys) (free tier works) — `LLM_PROVIDER=groq` is the default. For real patient data use a self-hosted model instead (`LLM_PROVIDER=openai_compatible`, see [docs/deployment.md](docs/deployment.md#4-the-model-keep-patient-data-on-your-machines-decision-required)).

## One-time setup

```bash
git clone <this-repo>
cd medagent

uv sync                          # installs Python deps into .venv
cp .env.example .env              # fill in GROQ_API_KEY, and pick an ADMIN_BOOTSTRAP_PASSWORD
./scripts/setup_vendor_mcp.sh     # clones + builds healthcare-mcp and med-research-mcp-suite into vendor/
                                   # (medical-mcp needs no setup -- it's a public npx package)
```

At minimum, edit `.env` and set:

```bash
GROQ_API_KEY=gsk_...
ADMIN_BOOTSTRAP_PASSWORD=some-password-you-choose
```

`ADMIN_BOOTSTRAP_USERNAME`/`ADMIN_BOOTSTRAP_PASSWORD` seed exactly one admin account the first time the app starts against an empty database — there's no public registration endpoint, so every other account is created by an admin via `POST /admin/users`.

---

## Option A: Hybrid local dev

Backing services in Docker, the app itself bare on the host for fast-reload development.

```bash
# 1. Start Postgres (with pgvector) and the 3 MCP servers (published on host ports)
make docker-up-deps

# 2. Point .env at their host-port addresses instead of the container-DNS
#    defaults -- uncomment this block in .env (already present, commented out):
#      MEDICAL_MCP_URL=http://localhost:8081
#      HEALTHCARE_MCP_URL=http://localhost:3001
#      RESEARCH_MCP_URL=http://localhost:3002
#      POSTGRES_DSN=postgresql://medagent:medagent@localhost:5432/medagent

# 3. Apply migrations
make db-upgrade

# 4. Run the app
make serve
```

The API is now at `http://localhost:8000`. Tear down the backing services with `docker compose down` when done.

---

## Option B: Full Docker stack

Everything — Postgres (with pgvector), all 3 MCP servers, and the app — runs in containers.

```bash
make docker-up
```

This builds and starts 8 services: `postgres` (the `pgvector/pgvector` image), `healthcare-mcp`, `research-mcp`, `medical-mcp-bridge`, `medagent`, `prometheus`, `grafana`, `tempo`. The app runs its own `alembic upgrade head` on startup. Give it a minute — `medagent`'s healthcheck has a 120s start period since it loads a local sentence-transformers model on boot.

```bash
docker compose ps        # check all services are "healthy"
make docker-logs         # tail logs from everything
make docker-down         # tear down
```

If port 5432 is already taken by a local Postgres install, start with `POSTGRES_HOST_PORT=5433 docker compose up -d --build` instead.

The API is at `http://localhost:8000` either way. Grafana is at `http://localhost:3000` (login `admin` / your `GRAFANA_ADMIN_PASSWORD`), Prometheus at `http://localhost:9090`.

### Observability

The provisioned **MedAgent Overview** dashboard shows, from `/metrics`: HTTP rate and latency, LLM calls by outcome (success / rate_limited / timeout / server_error / client_error / circuit_open), LLM latency and token spend, retries by reason, circuit-breaker state per dependency (Groq, each MCP server), MCP requests by outcome, specialist-agent steps, the evidence agent's ReAct-vs-fallback split (the number to watch for the agent's health), ReAct tool calls, guardrail findings and context truncations. All labels come from small fixed sets, so cardinality stays bounded.

**Traces.** With the Docker stack, spans go to Grafana Tempo. In Grafana (`http://localhost:3000`) open *Explore* -> *Tempo* and run `{ resource.service.name = "medagent" }` (or use the *Recent Traces* panel). One trace shows the request, each graph node, ReAct turns, LLM calls, tool calls and MCP calls, with retries as events. Every log line carries `trace_id` and `request_id` (also returned as the `X-Request-ID` header). Traces, logs and metrics contain no patient data. Set `OTEL_CONSOLE_EXPORTER=true` for local span printing, or `OTEL_SAMPLE_RATIO` to sample.

**Security defaults.**
- Login is limited per username (5/min) and per client IP (20/min); agent runs (`/assess`, `/followup`, `/drug-check`, WebSocket drafts) are limited per user (6/min, burst 3). Over the limit: `429` with `Retry-After`. Limits are per process, or shared by every replica with `STATE_BACKEND=postgres` (`LOGIN_ATTEMPTS_*`, `AGENT_RUNS_*`).
- Request bodies over 1 MB get `413`; text fields and lists have maximum sizes (messages up to 4000 characters, patient ids `[A-Za-z0-9_-]{1,64}`); passwords need 12+ characters; roles are `doctor` or `admin`.
- Every response has `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` and `Cache-Control: no-store`.
- `APP_ENV=production` refuses to start with the placeholder or a short `JWT_SECRET_KEY` (32+ chars, e.g. `openssl rand -hex 32`) or a short bootstrap password, hides `/docs`, and adds HSTS. In development the app logs a warning when the placeholder secret is used.
- Docker Compose publishes every port on `127.0.0.1` only, and the app container runs as a non-root user.
- `make deps-audit` scans the locked Python dependencies and `make mcp-audit` the three MCP servers' npm lockfiles for known vulnerabilities (both in CI). The vendored MCP servers are pinned to fixed commits.
- Search terms sent to external medical APIs are stripped of the patient's name, record ID, dates and numbers (`outbound_search_query`); [docs/deployment.md §11](docs/deployment.md#11-outbound-network-traffic) lists every outbound host.
- **Patient data and the model:** `LLM_PROVIDER=openai_compatible` with `LLM_BASE_URL`/`LLM_MODEL` uses a model you host (Ollama, vLLM, LM Studio), so prompts never leave your machines. Production refuses Groq or a public-host model unless `ALLOW_EXTERNAL_LLM_WITH_PHI=true` (you have a data agreement).
- **Accounts:** `GET /admin/users`, `POST /admin/users/{username}/disable|enable|reset-password|revoke-tokens`, `POST /auth/change-password`. Each of these except enable ends that user's existing sessions immediately. All are audited.
- **Retention:** `make purge-expired` (dry run) / `python -m medagent.cli purge-expired --execute`, only when `RETENTION_DAYS` / `AUDIT_RETENTION_DAYS` are set.
- **Production:** `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build` — Caddy with automatic HTTPS as the only public service, required secrets, optional Ollama. Full runbook (secrets, model choice, encrypted disks and backups, upgrades, retention schedule, accounts): [docs/deployment.md](docs/deployment.md).
- **Kubernetes / several replicas:** `deploy/k8s/` (Kustomize). `STATE_BACKEND=postgres` shares drafts and abuse limits through the database; `make k8s-local-up` runs 3 replicas on kind; `overlays/aws` targets EKS + RDS + ALB ([docs/deployment.md](docs/deployment.md#10-kubernetes-and-aws-several-replicas)).
- **AWS infrastructure:** `deploy/terraform/` creates the VPC, EKS, RDS, ECR, the certificate and the secrets (`make tf-apply`), and `make aws-overlay` turns its outputs into a ready overlay. Steps and costs are in [deploy/terraform/README.md](deploy/terraform/README.md).

**Prompt tracing (Langfuse, optional).** Tempo shows timings and outcomes but never prompt text. To see what each agent sent the model and what it answered, create a project at [cloud.langfuse.com](https://cloud.langfuse.com) (or self-host Langfuse) and set in `.env`:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or https://us.cloud.langfuse.com
```

Each request then appears in Langfuse as one trace (route → graph nodes → ReAct turns → LLM calls), every LLM call with its messages, answer, tool calls and tokens, under the same `trace_id` as in Tempo. `make eval` runs are sent too. Prompts go to Langfuse only, never to Tempo or the logs.

**Privacy note:** prompts contain patient data. Use a hosted Langfuse only with synthetic data; production refuses a non-local `LANGFUSE_BASE_URL` unless `ALLOW_PROMPT_TRACING_WITH_PHI=true` (you have a data agreement).

**Readiness.** `GET /ready` (no auth) reports `ok` / `degraded` / `down` per dependency; 503 only when Postgres is down. `/health` is liveness only.

Reliability policy (tune in `.env`): a permanent error (bad model name, bad request) is attempted once; timeouts, connection failures and 5xx are retried with backoff (3 attempts, `RETRY_BUDGET_SECONDS` total); a rate limit is waited out only if it asks for `RETRY_MAX_WAIT_SECONDS` or less, otherwise the request fails fast; after 5 consecutive upstream failures the breaker opens and calls fail immediately for 30 s. When the LLM or an MCP server is down the report degrades (see *Partial failure* below) instead of hanging.

---

## Walkthrough: curl commands

Works identically against both options — same API, same port. Uses `python3 -c` instead of `jq` to extract JSON fields (swap in `jq` if you have it installed).

### 1. Log in as the bootstrapped admin

```bash
BASE=http://localhost:8000

ADMIN_TOKEN=$(curl -s -X POST "$BASE/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=$ADMIN_BOOTSTRAP_USERNAME&password=$ADMIN_BOOTSTRAP_PASSWORD" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Admin token: ${ADMIN_TOKEN:0:20}..."
```

### 2. Create a doctor account and log in as them

There's no self-registration — every account after the bootstrapped admin is created by an admin:

```bash
curl -s -X POST "$BASE/admin/users" \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"username": "dr.alice", "password": "alice-pass-2026", "role": "doctor"}'

ALICE_TOKEN=$(curl -s -X POST "$BASE/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=dr.alice&password=alice-pass-2026" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

### 3. Create a synthetic patient

`doctor_id` is stamped server-side from the JWT — never something you pass in the body. Posting the same id again updates the record; if the id belongs to another doctor's patient the request gets a generic `409` (the record is never overwritten, and its existence is not revealed). Conditions, allergies, and lab results all feed into the agents' reasoning, not just the medications (see [step 5b](#5b-confirm-allergy-conflict-detection)) — include them to see the full effect.

```bash
curl -s -X POST "$BASE/patients" \
  -H "Authorization: Bearer $ALICE_TOKEN" -H "Content-Type: application/json" \
  -d '{
    "id": "P-TEST-001",
    "name": "Patient Alpha",
    "age": 68,
    "sex": "F",
    "conditions": ["type 2 diabetes", "hypertension"],
    "allergies": ["Sulfa"],
    "medications": [{"name": "Metformin"}, {"name": "Glimepiride"}],
    "lab_results": [{
      "test_name": "HbA1c", "value": 9.8, "unit": "%",
      "reference_low": 4.0, "reference_high": 5.7,
      "collected_at": "2026-01-01T00:00:00Z"
    }]
  }' | python3 -m json.tool
```

### 4. Run a full assessment (LangGraph: triage → drug-safety + evidence + trial-finder in parallel → report)

Evidence lookup is tailored to the patient's recorded conditions/allergies (not just the message), and the report always includes deterministic **Known Allergies** / **Lab Flags** (abnormal-only) sections regardless of which agents ran, driven straight from the record.

```bash
curl -s -X POST "$BASE/patients/P-TEST-001/assess" \
  -H "Authorization: Bearer $ALICE_TOKEN" -H "Content-Type: application/json" \
  -d '{"message": "Check interactions for current medications and any relevant treatment evidence"}' \
  | python3 -m json.tool
```

### 5. Check a proposed new drug against current medications

```bash
curl -s -X POST "$BASE/drug-check" \
  -H "Authorization: Bearer $ALICE_TOKEN" -H "Content-Type: application/json" \
  -d '{"patient_id": "P-TEST-001", "new_drug": "Ibuprofen"}' \
  | python3 -m json.tool
```

### 5b. Confirm allergy conflict detection

Checking a drug that matches a recorded allergy (case-insensitive substring match) always appends a deterministic `⚠️ ALLERGY CONFLICT` line to the answer — never left solely to the LLM to notice, even though it usually does mention it too:

```bash
curl -s -X POST "$BASE/drug-check" \
  -H "Authorization: Bearer $ALICE_TOKEN" -H "Content-Type: application/json" \
  -d '{"patient_id": "P-TEST-001", "new_drug": "Sulfamethoxazole"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['answer'])"
```

### 6. Follow-up visit (recalls the patient, re-runs relevant agents)

History is kept, not overwritten: a stopped medication keeps its row (`status: stopped`), a dose change closes the old row and opens a new one, and lab results accumulate (the record shows the latest per test). Each assessment saves a visit with a snapshot of the record, so the next report gets a deterministic **Changes Since Last Visit** section (medications started/stopped/changed, new conditions or allergies, newer labs). To see it, change the record after step 4 — here Glimepiride is stopped, Empagliflozin started, and a newer HbA1c added:

```bash
curl -s -X POST "$BASE/patients" \
  -H "Authorization: Bearer $ALICE_TOKEN" -H "Content-Type: application/json" \
  -d '{
    "id": "P-TEST-001", "name": "Patient Alpha", "age": 68, "sex": "F",
    "conditions": ["type 2 diabetes", "hypertension"], "allergies": ["Sulfa"],
    "medications": [{"name": "Metformin"}, {"name": "Empagliflozin"}],
    "lab_results": [{
      "test_name": "HbA1c", "value": 8.4, "unit": "%",
      "reference_low": 4.0, "reference_high": 5.7,
      "collected_at": "2026-04-01T00:00:00Z"
    }]
  }' > /dev/null
```

Then run the follow-up:

```bash
curl -s -X POST "$BASE/patients/P-TEST-001/followup" \
  -H "Authorization: Bearer $ALICE_TOKEN" -H "Content-Type: application/json" \
  -d '{"message": "Any updates given her latest labs?"}' \
  | python3 -m json.tool
```

### 7. Audit log for a patient

```bash
curl -s "$BASE/patients/P-TEST-001/audit-log?limit=50&offset=0" -H "Authorization: Bearer $ALICE_TOKEN" | python3 -m json.tool
```

Newest first, paged with `limit` (1–500, default 100) and `offset`.

### 7b. Interaction-check history

Every interaction check (`/drug-check`, and assessments or follow-ups where the Drug Safety agent ran) is recorded, one row per drug pair with severity, description and source. Over the WebSocket, only an approved or edited report records its checks.

```bash
curl -s "$BASE/patients/P-TEST-001/interactions?limit=50" -H "Authorization: Bearer $ALICE_TOKEN" | python3 -m json.tool
```

Newest first, same paging and ownership rules as the audit log.

### 8. Confirm per-doctor data isolation

A second doctor gets a plain 404 for a patient they don't own — not 403, so patient existence is never leaked across doctors. Admins bypass isolation and can see any patient.

```bash
curl -s -X POST "$BASE/admin/users" \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"username": "dr.bob", "password": "bob-pass-2026-x", "role": "doctor"}'

BOB_TOKEN=$(curl -s -X POST "$BASE/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=dr.bob&password=bob-pass-2026-x" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Bob fetching Alice's patient (expect 404):"
curl -s -o /dev/null -w "%{http_code}\n" "$BASE/patients/P-TEST-001" -H "Authorization: Bearer $BOB_TOKEN"

echo "Admin fetching Alice's patient (expect 200):"
curl -s -o /dev/null -w "%{http_code}\n" "$BASE/patients/P-TEST-001" -H "Authorization: Bearer $ADMIN_TOKEN"
```

### 9. Health, metrics

```bash
curl -s "$BASE/health"
curl -s "$BASE/ready"
curl -s "$BASE/metrics" | grep medagent
```

### Human-in-the-loop chat (WebSocket)

`curl` can't speak WebSocket — use `websocat` or `wscat` (`npm i -g wscat`):

```bash
wscat -c "ws://localhost:8000/ws/P-TEST-001"
# first message must authenticate (the token is never put in the URL, where
# proxies and load balancers log it; the socket closes after 10 s without it):
#   {"type":"auth","token":"<paste $ALICE_TOKEN>"}
# type a message, e.g.: Any changes given her latest labs?
# server streams progress as it works:
#   {"type":"progress","step":"triage","status":"completed"}
#   {"type":"progress","step":"evidence","status":"completed"}   (or "failed" / "skipped")
#   {"type":"progress","step":"report","status":"completed"}
# then: {"type":"pending_approval","report":"..."}
# reply with: approve   (or "reject", or your own edited text to save instead)
```

A drafted report survives a dropped connection. Reconnect with `?resume=1` to get it back
(`"resumed": true`) and finish deciding:

```bash
wscat -c "ws://localhost:8000/ws/P-TEST-001?resume=1"
# then authenticate as above: {"type":"auth","token":"<paste $ALICE_TOKEN>"}
# server replies: {"type":"pending_approval","report":"...","resumed":true}
```

Connecting *without* `resume=1` discards any stale draft, so a fresh question is never mistaken for the reply to an old one. Drafts contain patient data: they are private to the doctor who drafted them and purged after `APPROVAL_DRAFT_TTL_MINUTES` (default 30). With the default `STATE_BACKEND=memory` they live in server memory and a restart loses them; with `postgres` they survive restarts and can be resumed on any replica.

### Guideline RAG (optional)

The evidence agent's `search_guidelines` tool searches guidelines you ingest into Postgres with pgvector (`.pdf` or `.txt`). The chunker splits on section headers (markdown, numbered like `3.1.2 Title`, or all-caps multi-word lines), then sizes each chunk to the embedding model's token limit, adds a `<source> › <section>` context prefix for embedding, and drops reference-list chunks. Each document is versioned (text hash, embedding model, chunker version): ingesting an unchanged document does nothing, a changed one replaces its chunks in one transaction, and `--force` re-ingests. Guidelines ingested into the old ChromaDB store must be ingested again. An optional cross-encoder re-ranker (`RAG_RERANKER_MODEL`, off by default for CPU latency) raised the share of answers reaching the model from 57 % to 76 %. `make rag-eval` measures retrieval quality over your PDFs ([docs/info.md §13.1](docs/info.md#131-measuring-retrieval-make-rag-eval)). A synthetic demo guideline is included:

```bash
make ingest-guideline file=docs/demo/synthetic-t2dm-guideline.txt source=synthetic-t2dm
```

With the full Docker stack, run it inside the container instead: `docker compose exec medagent python -m medagent.cli ingest-guideline --file ... --source ...`. Only ingest documents you are licensed to use.

---

## Safety and context controls

The LLM only ever summarizes; safety-relevant facts are computed deterministically and never left to it.

- **ReAct evidence agent**: the evidence agent runs a real ReAct loop (think → call a tool → read the result → repeat) over two tools, `search_medical_literature` and `search_guidelines`, so the model chooses what to search and with which terms. The loop is bounded (3 turns, a whole-loop deadline `AGENT_LOOP_TIMEOUT_SECONDS`, a prompt-size ceiling, capped and delimited tool output). An identical repeated search is never re-run, and a turn that makes no progress makes the model answer at once; a turn's tool calls run in parallel. If the model answers without retrieving anything, exhausts the loop, or the provider rejects a tool call, it falls back to the fixed retrieve-then-summarize pipeline, so a doctor always gets a grounded answer. The other agents stay fixed pipelines on purpose — drug-interaction and allergy checks are deterministic and never left to model judgment. Tradeoff: a ReAct run makes 2–3 LLM calls and spends roughly 2× the tokens of the fixed pipeline, which matters on small per-minute token tiers.

- **Prompt-injection hardening**: the doctor's message, patient-record fields (conditions, allergies, visit history — these are stored and replayed into every later prompt) and external MCP text are wrapped in `<<< >>>` delimiters, and each agent's system prompt says to treat that text as data, not instructions.
- **Output allergy cross-check**: on top of the input-side allergy check, the LLM's own response is scanned for any recorded allergy, so an alternative drug it suggests on its own is still flagged with `⚠️ ALLERGY CONFLICT`. A cheap sanity pass (empty response, system-prompt leakage) is logged, not auto-blocked.
- **Context budgets**: raw MCP payloads are curated to the clinically relevant fields (not dumped as JSON), every external text field and the final prompt are capped, and each past visit is truncated when replayed as history. Sizes are the `AGENT_*_TOKENS` settings in `.env.example` — tune them to your Groq tier's per-request limit (the free `openai/gpt-oss-20b` tier is 8000 tokens).
- **Figure check**: every evidence answer is checked for doses, percentages and lab values that appear in none of the sources the model was given (retrieved evidence, the patient record, the doctor's question). Unmatched figures get an `UNVERIFIED FIGURES` warning. Deterministic and advisory (no extra LLM call, never blocks); it checks numbers, not whether the reasoning follows.
- **Per-run token budget**: one assess/followup run stops launching further specialist agents once `AGENT_MAX_TOKENS_PER_RUN` is spent, and the report says which were skipped. Each step's cost and outcome is logged (`agent_step_completed` / `agent_step_skipped`).
- **Human approval**: the WebSocket chat drafts a report and saves nothing until the doctor approves, edits or rejects it. The pause is a LangGraph `interrupt()` held by a LangGraph checkpointer (in memory, or Postgres for several replicas), so a dropped connection can resume the draft. The REST endpoints save immediately.
- **Partial failure**: if one specialist agent fails (LLM or MCP outage), the report still ships with the other sections and the deterministic allergy/lab sections, plus a note saying what was unavailable (in fixed, doctor-safe wording, never the raw error). If *every* specialist fails there is nothing clinical to report and the request returns 502 -- or, if they were all rate-limited by the LLM/MCP provider, **429 with a `Retry-After` header** (and a `retry_after` field) and wording that says how long to wait, so a client can back off instead of treating it as an outage. Upstream errors never return their raw message (it can carry account ids or internal hostnames), only fixed text.

## Tests

```bash
make unit-tests    # network-free, requires Docker for Postgres-backed tests (auto-skips if unavailable)
make type-check    # mypy over src/medagent
make eval          # live evals: agent-eval gates + retrieval floors (real Groq + real MCP servers; needs GROQ_API_KEY)
make agent-eval    # agent eval report: tool use, errors, hallucination proxies, safety, cost per category; saved to eval-results/
make integration   # real embeddings + Postgres + medical-mcp bridge (skips what is unavailable)
make import-check  # layer rules (import-linter, contracts in pyproject.toml)
make deps-audit    # known-vulnerability scan of the locked dependencies (pip-audit)
make mcp-audit     # npm audit over the MCP servers' committed lockfiles
make pre-commit    # format + lint + type-check + import-check + unit-tests, run this before committing
```

`make eval` is the only check that exercises real model output and real MCP response shapes; the unit tests use mocks. It has caught bugs the unit tests could not (a severity parser that silently returned "none" on real data).

**Agent eval** (`make agent-eval`, [docs/info.md §17.1](docs/info.md#171-agent-eval-make-agent-eval)): 45 synthetic cases (interactions, allergy, kidney disease, evidence, history, trials, prompt injection, follow-up with approval, drug-check, multi-specialist and degraded reports) run end to end through the real assessment, follow-up and drug-check workflows; `make agent-eval` repeats safety categories three times, CI runs each case once. It reports tool-use rate, tool-call validity and errors, fallback rate, unsupported figures and fabricated references, expected-fact recall, the safety gates, triage accuracy, report completeness, follow-up changes and approval, degraded reports, and tokens / calls / seconds per run, and compares with the previous run. From the host, point the MCP URLs at `localhost` (see [docs/running.md](docs/running.md)).

### CI

- **Every push, any branch**: format check, lint, mypy, import-check, unit tests (`make ci-check`), and the dependency audits (`make deps-audit`, `make mcp-audit`).
- **Pull requests into `main` only**: additionally the golden-dataset eval against live MCP containers and a Docker build of every image. These are gated to PRs because they call the real Groq API and build several images. The eval job needs a `GROQ_API_KEY` repository secret (Settings → Secrets and variables → Actions).

## Docker image

`torch` is pinned to PyTorch's CPU-only wheels (`[tool.uv.sources]` in `pyproject.toml`). The default wheel bundles ~2.5 GB of NVIDIA CUDA libraries this CPU-only service never uses; pinning it took the image to ~2.1 GB. If you later deploy on GPU hardware, remove that override and re-lock.

## More detail

- [`docs/info.md`](docs/info.md): complete guide to the concepts and flows (workflow, agents, MCP, harness, observability, memory, security).
- [`docs/code-walkthrough.md`](docs/code-walkthrough.md): how the code runs (entry points, every request flow, where output goes) and what each file does.
- [`docs/running.md`](docs/running.md): commands for each stage, local → Docker → kind → AWS.
- [`docs/deployment.md`](docs/deployment.md): production runbook.
- [`AGENTS.md`](AGENTS.md): architecture, phase breakdown, dependency rules, and the MCP server contracts.
- [`docs/glossary.md`](docs/glossary.md): canonical terms.
