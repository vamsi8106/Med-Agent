# MedAgent

Clinical decision support agent for doctors. Checks drug interactions, retrieves treatment evidence, finds clinical trials, tracks patient history across sessions. All medical data from free open APIs via MCP servers. Patient data stays local (SQLite + ChromaDB). Synthetic data only for demos.

**Stack:** Python 3.12+, uv, ruff, pytest, pydantic v2, structlog, httpx, FastAPI.
**LLM:** Groq (initial) → swappable via `LLM_PROVIDER` env var. No Groq imports outside `llm/groq_provider.py`.
**MCP:** medical-mcp (FDA/WHO/RxNorm/PubMed), healthcare-mcp (ICD-10/trials/calculator), med-research-mcp-suite (cross-DB analysis). All run locally, no API keys.
**Storage:** SQLite (patients, meds, labs, visits), ChromaDB (guideline embeddings), sentence-transformers `all-MiniLM-L6-v2` (local).

# User Flows

**New patient:** Doctor enters demographics + meds + labs → Triage routes to Drug Safety Agent (checks interactions via MCP) + Evidence Agent (searches PubMed + RAG guidelines) in parallel → Report Agent synthesizes → saves patient to SQLite.

**Follow-up:** Doctor references patient ID → agent loads full history from SQLite → checks what changed → runs relevant agents → updates records.

**Drug interaction:** Doctor asks "can I add Drug X?" → loads current meds from memory → runs all pairwise interaction checks via MCP → flags risks with severity + FDA data.

**Trial search:** Doctor asks about trials → Trial Finder searches ClinicalTrials.gov via MCP → filters by condition/phase/status → cross-references PubMed for results.

**Evidence lookup:** Doctor asks clinical question → RAG retrieves from ingested guidelines → PubMed MCP fetches recent papers → structures response with citations.

# Structure

```
src/medagent/
├── core/                        # Phase 1 — shared kernel, zero external deps beyond pydantic
│   ├── interfaces.py            # ABCs: BaseLLMProvider, BaseTool, BaseAgent, BaseMemory
│   ├── models.py                # PatientContext, Medication, LabResult, DrugInteraction, Message, ToolCall, LLMResponse
│   ├── exceptions.py            # MedAgentError → ProviderError, ToolError, MemoryError, MCPError
│   ├── config.py                # Settings (pydantic-settings)
│   └── types.py                 # Enums: EvidenceGrade, InteractionSeverity, CKDStage, TrialPhase, AgentRole
├── llm/                         # Phase 1 — provider layer
│   ├── registry.py              # ProviderRegistry: get_provider(name) → BaseLLMProvider
│   ├── groq_provider.py         # Groq implementation
│   └── mock_provider.py         # deterministic, for tests
├── infra/                       # Phase 1+ — cross-cutting
│   ├── logging.py               # structlog JSON setup
│   ├── retry.py                 # @retry with exponential backoff + jitter
│   ├── rate_limiter.py          # token-bucket per API source
│   └── circuit_breaker.py       # degrade if MCP server down (Phase 6)
├── tools/                       # Phase 2
│   ├── base.py                  # BaseTool ABC + ToolResult
│   ├── registry.py              # ToolRegistry: auto-discovery
│   ├── decorators.py            # @tool: auto JSON schema from type hints
│   ├── mcp/                     # MCP clients (medical.py, healthcare.py, research.py)
│   └── custom/                  # interaction_checker, patient_context, lab_interpreter,
│                                # report_generator, guideline_retriever, dosage_calculator
├── agents/                      # Phase 3 (base + drug_safety), Phase 5 (rest)
│   ├── base.py                  # ReAct loop: perceive → think → act
│   ├── triage_agent.py          # routes to specialists
│   ├── drug_safety_agent.py     # interactions, adverse events
│   ├── evidence_agent.py        # literature + guideline retrieval
│   ├── trial_finder_agent.py    # ClinicalTrials.gov search
│   └── report_agent.py          # synthesizes multi-agent output
├── memory/                      # Phase 4
│   ├── patient_store.py         # SQLite CRUD for patient records
│   ├── session.py               # sliding-window conversation buffer
│   └── persistent.py            # SQLite schema + migrations
├── rag/                         # Phase 4
│   ├── embeddings.py            # sentence-transformers (local)
│   ├── vector_store.py          # ChromaDB adapter
│   ├── chunker.py               # section-header-aware, 512 tokens, 64 overlap
│   └── pipeline.py              # PDF → chunk → embed → store
├── workflows/                   # Phase 5
│   ├── patient_assessment.py    # triage → parallel(drug_safety, evidence) → report
│   ├── followup.py              # recall → check changes → advise → update
│   └── drug_check.py            # enumerate pairs → check → aggregate
└── app.py                       # Phase 6 — FastAPI + WebSocket
```

`tests/` mirrors `src/` 1:1. `docs/glossary.md` for canonical terms. `docs/adr/` for architecture decisions.

# Dependency Rules

```
core/     → nothing
infra/    → nothing (3rd-party only)
llm/      → core/
tools/    → core/, infra/
memory/   → core/
rag/      → core/
agents/   → core/, llm/, tools/, memory/, rag/
workflows/→ core/, agents/
app.py    → everything
```

Never import upward. `core/` never imports from `llm/`. `tools/` never imports from `agents/`.

# Phases

**Phase 1 — Foundation:** `core/` (all files) + `infra/` (logging, retry, rate_limiter) + `llm/` (groq, mock, registry). Unit tests for models, config, retry, mock provider. Gate: `make pre-commit` green.

**Phase 2 — Tools & MCP:** `tools/` (base, registry, decorators, mcp clients, interaction_checker, patient_context). Integration test with real medical-mcp. Gate: `search-drugs("metformin")` returns structured response via MCP.

**Phase 3 — Single Agent:** `agents/base.py` (ReAct loop) + `drug_safety_agent` + `memory/session.py`. Gate: "Check interactions for Metformin + Glimepiride" → correct response with citations.

**Phase 4 — Memory & RAG:** `memory/` (patient_store, persistent with SQLite) + `rag/` (full pipeline) + `guideline_retriever` + `lab_interpreter`. Gate: follow-up visit recalls patient, retrieves matching guidelines.

**Phase 5 — Multi-Agent:** remaining agents + `workflows/` + `report_generator` + `dosage_calculator`. Gate: complex patient → multi-agent → structured report.

**Phase 6 — Production:** `app.py` (FastAPI + WebSocket) + `infra/` (tracing, metrics, circuit_breaker, middleware) + Docker. Gate: `docker compose up` runs full system.

# SQLite Schema

```sql
patients    (id, name, age, sex, weight_kg, height_cm, conditions JSON, allergies JSON, created_at, updated_at)
medications (id, patient_id FK, name, brand_name, dose, frequency, route, start_date, end_date, status, created_at)
lab_results (id, patient_id FK, test_name, value, unit, reference_low, reference_high, is_abnormal, collected_at)
visits      (id, patient_id FK, visit_date, chief_complaint, assessment, plan, agent_session_id, created_at)
interactions_log (id, patient_id FK, drug_a, drug_b, severity, description, source, checked_at)
```

# MCP Server Config

```yaml
medical-mcp:     { command: npx, args: ["-y", "medical-mcp"], transport: stdio }
healthcare-mcp:  { command: node, args: ["./vendor/healthcare-mcp/build/index.js"], transport: stdio }
med-research:    { command: node, args: ["./vendor/med-research-mcp-suite/dist/index.js"], transport: stdio }
```

Tools available: `search-drugs`, `get-drug-details`, `search-drug-nomenclature`, `get-health-statistics`, `search-medical-literature`, `get-article-details`, `search-clinical-guidelines`, `fda_drug_lookup`, `clinical_trials_search`, `medical_terminology` (ICD-10), `medical_calculator`, `comprehensive-analysis`, `drug-safety-profile`.

# Custom Tools

| Tool | Does | Inputs → Outputs |
|---|---|---|
| `interaction_checker` | pairwise drug checks via MCP | `list[Medication]` → `list[DrugInteraction]` |
| `patient_context` | CRUD patient records | `patient_id` → `PatientContext` |
| `lab_interpreter` | flag abnormals by age/sex/condition | `list[LabResult], PatientContext` → `list[LabFlag]` |
| `report_generator` | structure findings with citations | `AgentResult` → markdown report |
| `guideline_retriever` | RAG over ingested guidelines | `query, top_k` → `list[ClinicalEvidence]` |
| `dosage_calculator` | renal/hepatic dose adjustment | `Medication, PatientContext` → adjusted dose |

# Rules

- Type-annotate everything including `-> None`.
- Async for I/O, sync for CPU.
- `structlog` logger, never `print()`.
- Datetimes UTC, timezone-aware. No naive `datetime`.
- Raise `MedAgentError` subclasses, never bare `Exception`.
- Config via `Settings` singleton. New env var → `.env.example` + `config.py`.
- Synthetic patient data only. Names like "Patient Alpha", IDs like "P-TEST-001".
- No abstraction without a second caller.
- Test with `MockLLMProvider`. No network in unit tests.

# Commands

```
make install         # uv sync
make test            # uv run pytest
make unit-tests      # uv run pytest tests/unit/
make lint-check      # uv run ruff check .
make format-fix      # uv run ruff format .
make pre-commit      # format-fix + lint-fix + lint-check + unit-tests
make serve           # uv run uvicorn medagent.app:app --reload
```
