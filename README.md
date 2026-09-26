# MedAgent

Clinical decision-support agent for doctors: checks drug interactions (including against recorded allergies), retrieves treatment evidence tailored to the patient's conditions, finds clinical trials, and flags abnormal labs — with each doctor scoped to only their own patients. Built on LangGraph, FastAPI, Postgres, and ChromaDB, with all medical data sourced from free MCP servers (FDA/WHO/RxNorm/PubMed, ICD-10/trials/calculators, cross-database analysis).

Two ways to run it locally:

- **[Option A — Hybrid local dev](#option-a-hybrid-local-dev)**: the app runs bare on your host (fast reload via `uvicorn --reload`), backing services (Postgres, ChromaDB, the 3 MCP servers) run in Docker.
- **[Option B — Full Docker stack](#option-b-full-docker-stack)**: everything, including the app itself, runs in Docker — closest to the production topology.

Both are exercised with the same `curl` walkthrough below.

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose
- Node.js/npm (only needed once, to vendor the two MCP servers built from source)
- A [Groq API key](https://console.groq.com/keys) (free tier works) — `LLM_PROVIDER=groq` is the default

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
# 1. Start Postgres, ChromaDB, and the 3 MCP servers (published on host ports)
make docker-up-deps

# 2. Point .env at their host-port addresses instead of the container-DNS
#    defaults -- uncomment this block in .env (already present, commented out):
#      MEDICAL_MCP_URL=http://localhost:8081
#      HEALTHCARE_MCP_URL=http://localhost:3001
#      RESEARCH_MCP_URL=http://localhost:3002
#      CHROMA_HOST=localhost
#      CHROMA_PORT=8001
#      POSTGRES_DSN=postgresql://medagent:medagent@localhost:5432/medagent

# 3. Apply migrations
make db-upgrade

# 4. Run the app
make serve
```

The API is now at `http://localhost:8000`. Tear down the backing services with `docker compose down` when done.

---

## Option B: Full Docker stack

Everything — Postgres, ChromaDB, all 3 MCP servers, and the app — runs in containers.

```bash
make docker-up
```

This builds and starts 8 services: `postgres`, `chromadb`, `healthcare-mcp`, `research-mcp`, `medical-mcp-bridge`, `medagent`, `prometheus`, `grafana`. The app runs its own `alembic upgrade head` on startup. Give it a minute — `medagent`'s healthcheck has a 120s start period since it loads a local sentence-transformers model on boot.

```bash
docker compose ps        # check all services are "healthy"
make docker-logs         # tail logs from everything
make docker-down         # tear down
```

If port 5432 is already taken by a local Postgres install, start with `POSTGRES_HOST_PORT=5433 docker compose up -d --build` instead.

The API is at `http://localhost:8000` either way. Grafana is at `http://localhost:3000` (login `admin` / your `GRAFANA_ADMIN_PASSWORD`), Prometheus at `http://localhost:9090`.

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
  -d '{"username": "dr.alice", "password": "alicepass1", "role": "doctor"}'

ALICE_TOKEN=$(curl -s -X POST "$BASE/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=dr.alice&password=alicepass1" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

### 3. Create a synthetic patient

`doctor_id` is stamped server-side from the JWT — never something you pass in the body. Conditions, allergies, and lab results all feed into the agents' reasoning, not just the medications (see [step 5b](#5b-confirm-allergy-conflict-detection)) — include them to see the full effect.

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

```bash
curl -s -X POST "$BASE/patients/P-TEST-001/followup" \
  -H "Authorization: Bearer $ALICE_TOKEN" -H "Content-Type: application/json" \
  -d '{"message": "Any updates given her latest labs?"}' \
  | python3 -m json.tool
```

### 7. Audit log for a patient

```bash
curl -s "$BASE/patients/P-TEST-001/audit-log" -H "Authorization: Bearer $ALICE_TOKEN" | python3 -m json.tool
```

### 8. Confirm per-doctor data isolation

A second doctor gets a plain 404 for a patient they don't own — not 403, so patient existence is never leaked across doctors. Admins bypass isolation and can see any patient.

```bash
curl -s -X POST "$BASE/admin/users" \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"username": "dr.bob", "password": "bobpass123", "role": "doctor"}'

BOB_TOKEN=$(curl -s -X POST "$BASE/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=dr.bob&password=bobpass123" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Bob fetching Alice's patient (expect 404):"
curl -s -o /dev/null -w "%{http_code}\n" "$BASE/patients/P-TEST-001" -H "Authorization: Bearer $BOB_TOKEN"

echo "Admin fetching Alice's patient (expect 200):"
curl -s -o /dev/null -w "%{http_code}\n" "$BASE/patients/P-TEST-001" -H "Authorization: Bearer $ADMIN_TOKEN"
```

### 9. Health, metrics

```bash
curl -s "$BASE/health"
curl -s "$BASE/metrics" | grep medagent
```

### Human-in-the-loop chat (WebSocket)

`curl` can't speak WebSocket — use `websocat` or `wscat` (`npm i -g wscat`):

```bash
wscat -c "ws://localhost:8000/ws/P-TEST-001?token=$ALICE_TOKEN"
# type a message, e.g.: Any changes given her latest labs?
# server replies: {"type":"pending_approval","report":"..."}
# reply with: approve   (or "reject", or your own edited text to save instead)
```

---

## Safety and context controls

The LLM only ever summarizes; safety-relevant facts are computed deterministically and never left to it.

- **ReAct evidence agent**: the evidence agent runs a real ReAct loop (think → call a tool → read the result → repeat) over two tools, `search_medical_literature` and `search_guidelines`, so the model chooses what to search and with which terms. The loop is bounded (3 turns, a prompt-size ceiling, capped and delimited tool output; the last turn tells the model to write its answer). If the model answers without retrieving anything, exhausts the loop, or the provider rejects a tool call, it falls back to the fixed retrieve-then-summarize pipeline, so a doctor always gets a grounded answer. The other agents stay fixed pipelines on purpose — drug-interaction and allergy checks are deterministic and never left to model judgment. Tradeoff: a ReAct run makes 2–3 LLM calls and spends roughly 2× the tokens of the fixed pipeline, which matters on small per-minute token tiers.

- **Prompt-injection hardening**: the doctor's message, patient-record fields (conditions, allergies, visit history — these are stored and replayed into every later prompt) and external MCP text are wrapped in `<<< >>>` delimiters, and each agent's system prompt says to treat that text as data, not instructions.
- **Output allergy cross-check**: on top of the input-side allergy check, the LLM's own response is scanned for any recorded allergy, so an alternative drug it suggests on its own is still flagged with `⚠️ ALLERGY CONFLICT`. A cheap sanity pass (empty response, system-prompt leakage) is logged, not auto-blocked.
- **Context budgets**: raw MCP payloads are curated to the clinically relevant fields (not dumped as JSON), every external text field and the final prompt are capped, and each past visit is truncated when replayed as history. Sizes are the `AGENT_*_TOKENS` settings in `.env.example` — tune them to your Groq tier's per-request limit (the free `openai/gpt-oss-20b` tier is 8000 tokens).
- **Per-run token budget**: one assess/followup run stops launching further specialist agents once `AGENT_MAX_TOKENS_PER_RUN` is spent, and the report says which were skipped. Each step's cost and outcome is logged (`agent_step_completed` / `agent_step_skipped`).
- **Human approval**: the WebSocket chat drafts a report and saves nothing until the doctor approves, edits or rejects it. The REST endpoints save immediately.

## Tests

```bash
make unit-tests    # network-free, requires Docker for Postgres-backed tests (auto-skips if unavailable)
make type-check    # mypy over src/medagent
make eval          # golden-dataset eval against real Groq + real MCP servers (needs GROQ_API_KEY)
make pre-commit    # format + lint + type-check + unit-tests, run this before committing
```

`make eval` is the only check that exercises real model output and real MCP response shapes; the unit tests use mocks. It has caught bugs the unit tests could not (a severity parser that silently returned "none" on real data).

### CI

- **Every push, any branch**: format check, lint, mypy, unit tests (`make ci-check`).
- **Pull requests into `main` only**: additionally the golden-dataset eval against live MCP containers and a Docker build of every image. These are gated to PRs because they call the real Groq API and build several images. The eval job needs a `GROQ_API_KEY` repository secret (Settings → Secrets and variables → Actions).

## Docker image

`torch` is pinned to PyTorch's CPU-only wheels (`[tool.uv.sources]` in `pyproject.toml`). The default wheel bundles ~2.5 GB of NVIDIA CUDA libraries this CPU-only service never uses; pinning it took the image to ~2.1 GB. If you later deploy on GPU hardware, remove that override and re-lock.

## More detail

See `AGENTS.md` for architecture, phase breakdown, dependency rules, and the MCP server contracts.
