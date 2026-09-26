# Glossary

Canonical names for every domain concept in MedAgent. Use these exact terms in code, docs, conversations, and commit messages. Update this file in the same PR that introduces or renames a term.

---

## Clinical Domain

| Term | Definition |
|---|---|
| **Patient Context** | The complete profile of a patient: demographics, active medications, lab results, conditions, allergies, visit history. Stored in Postgres, loaded per request and handed to every agent. |
| **Medication** | A drug a patient is currently taking or has taken. Normalized to generic name via RxNorm. Tracks dose, frequency, route, status (active/discontinued/on_hold). |
| **Lab Result** | A single lab test value with units, reference range, and abnormal flag. Timestamped. Examples: HbA1c 8.9%, eGFR 52 mL/min/1.73m². |
| **Drug Interaction** | A clinically significant effect when two or more drugs are taken together. Graded by InteractionSeverity. Source-attributed (OpenFDA, medical-mcp). |
| **Evidence Grade** | Strength of clinical evidence supporting a recommendation. Enum: `A`, `B`, `C`, `D` (A strongest). Defined, and `ClinicalEvidence.grade` exists, but no agent or tool assigns it yet -- it is always `None`. |
| **Interaction Severity** | Risk level of a drug-drug interaction. Enum: `contraindicated`, `major`, `moderate`, `minor`, `none` (`none` = no severity signal found in the source response). Parsed from research-mcp's `riskProfile.level` (High/Medium/Low → major/moderate/minor). |
| **CKD Stage** | Chronic kidney disease stage derived from eGFR. Enum: `stage_1` (≥90), `stage_2` (60-89), `stage_3a` (45-59), `stage_3b` (30-44), `stage_4` (15-29), `stage_5` (<15). Defined but not used by any code yet. |
| **Trial Phase** | Clinical trial phase. Enum: `early_phase_1`, `phase_1`, `phase_2`, `phase_3`, `phase_4`, `not_applicable`. Defined but not used by any code yet. |
| **Clinical Evidence** | A piece of evidence retrieved from PubMed or RAG: source, title, summary, evidence grade, citation ID. |
| **Differential Diagnosis** | A ranked list of possible diagnoses given symptoms, history, and lab results. Not a definitive diagnosis. |
| **Chief Complaint** | The primary reason the patient is being seen. Free text recorded per visit. |
| **Reference Range** | Normal value boundaries for a lab test (`reference_low` / `reference_high`), supplied with each result. Not adjusted by age, sex, or conditions -- `lab_interpreter` compares each value to its own range only. |

## System Architecture

| Term | Definition |
|---|---|
| **LLM Provider** | An abstraction over a specific LLM API (Groq, Anthropic, OpenAI). Implements `BaseLLMProvider`. Swappable via `LLM_PROVIDER` config. |
| **Provider Registry** | Singleton that maps provider names to `BaseLLMProvider` implementations. `get_provider("groq")` returns the Groq provider instance. |
| **Mock Provider** | A deterministic `BaseLLMProvider` for testing. Returns canned responses. No network. No API key. |
| **Tool** | A callable capability an agent can invoke. Implements `BaseTool`: a `name`, a `description`, a JSON-schema `parameters` dict (for LLM function calling), and an async `run()` returning a `ToolResult`. |
| **Tool Registry** | Holds a set of tools. `register()`, `get(name)`, `list_tools()`, and `schemas()` (the OpenAI/Groq function-calling format sent to the LLM). One is built per ReAct run. |
| **@tool Decorator** | Decorator that converts a typed async function into a `BaseTool` with auto-generated JSON schema from type hints. |
| **MCP Client** | A client that reaches an MCP-adjacent server over HTTP (each server runs as its own container), calls its tools, and returns structured results. Only medical-mcp speaks the real MCP protocol under the hood, via a stdio<->HTTP bridge sidecar; healthcare-mcp and med-research-mcp-suite expose plain REST APIs. |
| **MCP Server** | A containerized Node.js process that exposes medical tools. MedAgent integrates three: medical-mcp, healthcare-mcp, med-research-mcp-suite -- see AGENTS.md's "MCP Server Config" for each one's real transport/contract. |

## Agent System

| Term | Definition |
|---|---|
| **Agent** | A specialist that answers one kind of question for a `PatientContext` + message. Implements `BaseAgent`. Most agents are **fixed pipelines** (deterministic tool calls, then one LLM summary); only the Evidence Agent is a **ReAct agent**. |
| **Agent Role** | The specialization of an agent. Enum: `triage`, `drug_safety`, `evidence`, `trial_finder`, `report`. |
| **Triage Agent** | The routing agent. Keyword-matches the doctor's query to pick specialist agents (defaults to Evidence). Uses no LLM and calls no tools. |
| **Drug Safety Agent** | Fixed pipeline: `interaction_checker` over every medication pair, a deterministic allergy check, then one LLM summary. Never a ReAct loop -- interaction and allergy checks must not depend on model judgment. |
| **Evidence Agent** | The ReAct agent. Chooses between two tools, `search_medical_literature` (PubMed via medical-mcp) and `search_guidelines` (RAG via `guideline_retriever`), then answers with citations. Falls back to a fixed retrieve-then-summarize pipeline when the loop can't produce a grounded answer. |
| **Trial Finder Agent** | Searches ClinicalTrials.gov for matching trials. Primary tool: `healthcare-mcp/clinical_trials_search`. |
| **Report Agent** | Synthesizes outputs from other agents into a structured clinical report with citations. |
| **Agent Context** | What every agent receives: the `PatientContext` and the doctor's message. (`ReActAgent` also holds a `SessionMemory` and a `ToolRegistry`.) |
| **Agent Result** | The structured output of one specialist run (`AgentResult`): role, summary text, evidence cited, interactions found, and token `usage`. No confidence score. |
| **ReAct Loop** | Reasoning + Acting pattern (`ReActAgent`, a LangGraph think → act graph). The model calls tools, reads the results, and repeats until it answers. Bounded by `max_iterations`, a whole-loop deadline, a prompt-size ceiling, duplicate-call detection, and a final-answer directive (on the last turn, or immediately after a stall). A turn's tool calls run concurrently; tool failures are returned to the model as error results, not raised. |

## Safety & Context Control

| Term | Definition |
|---|---|
| **Guardrails** | The input/output checks around every LLM call (`infra/guardrails.py`). The LLM only summarizes; safety-relevant facts are computed deterministically and never left to it. |
| **Untrusted Text** | Any text not written by this system: the doctor's message, stored patient fields (conditions, allergies, visit history -- replayed into every later prompt), and MCP/tool output. `wrap_untrusted()` delimits it with `<<< >>>` and system prompts tell the model to treat it as data, not instructions. |
| **Output Allergy Check** | Scans the LLM's own response for any recorded allergy and appends `ALLERGY CONFLICT` if found. Complements the input-side check, which only covers the drugs a caller explicitly asked about. |
| **Unverified Figures** | The warning appended to an evidence answer by the Figure Check, listing figures that appear in none of the sources the model was given. |
| **Output Validation** | Cheap sanity checks on an LLM response (empty/too short, apparent system-prompt leakage). Logged, never auto-blocked: withholding clinical content on a false positive is worse than showing an odd response. |
| **Context Budget** | Caps on how much text enters a prompt (`infra/context_budget.py`): per external field, per visit replayed as history, and per assembled prompt. Sized with `AGENT_*_TOKENS` settings. |
| **Token Estimate** | Tokens estimated as characters / 3.5 -- measured against real Groq usage (dense medical text is 3.9-5 chars/token). Counting whitespace words instead undercounts ~2x. |
| **Truncation** | `truncate_text()` cuts on a word boundary, appends a marker, and logs `context_truncated`, so lost context is visible rather than silent. |
| **Agent Run Tracker** | Per-run token budget (`AGENT_MAX_TOKENS_PER_RUN`) and step trace. Once spent, remaining specialist agents are skipped and the report says which. Best-effort: parallel agents can't see each other's spend. |
| **Loop Deadline** | A whole-loop time limit on a ReAct run (`AGENT_LOOP_TIMEOUT_SECONDS`). LLM and MCP timeouts are per call, so without it a hung provider (30s × retries) could hold a multi-turn loop for minutes. On expiry the agent falls back to its fixed pipeline. |
| **Duplicate Call** | A tool call identical to an earlier one in the same loop (same tool, arguments equal ignoring case/spacing/order). It is not re-run; the model is told it already has that result. |
| **Stall** | A ReAct turn in which every call was a duplicate or an error, i.e. no progress. The next turn asks the model for its answer immediately instead of waiting for the last allowed turn. |
| **Figure Check** | Deterministic check that every figure with a unit an evidence answer states ("850 mg", "45%") has its number in the retrieved evidence, the patient record, or the doctor's question. Unmatched figures get an advisory `UNVERIFIED FIGURES` warning; the answer is never blocked. Checks numbers only -- not units, not whether the reasoning follows. |
| **Deterministic Fallback** | The Evidence Agent's fixed retrieve-then-summarize pipeline, used when the ReAct loop answers without retrieving anything, exhausts its turns, or the provider rejects a tool call. An unsourced answer is never returned. |
| **Human-in-the-Loop Approval** | The WebSocket chat drafts a report but saves nothing until the doctor approves, edits, or rejects it. The pause is a LangGraph `interrupt()` in the follow-up graph. REST endpoints save immediately. |
| **Draft** | A report awaiting the doctor's decision. Held by the checkpointer under a `user_id:patient_id` thread, resumable after a dropped connection with `?resume=1`, deleted once decided, and purged after `APPROVAL_DRAFT_TTL_MINUTES`. |
| **Degraded Report** | A report produced when some specialist agents failed: the surviving sections plus the deterministic allergy/lab sections and a `## Note` saying what was unavailable, in fixed doctor-safe wording (never the raw error). If every specialist fails, `AllAgentsFailedError` (502) is raised instead. |
| **Per-Doctor Isolation** | Every patient-data row carries `doctor_id`, stamped server-side from the creating doctor's JWT and immutable. A doctor sees only their own patients (a non-owner gets 404, not 403); admins bypass it. |
| **Audit Log** | Durable record of every patient-data access (created, viewed, assessed, approved, ...), with the acting user. |

## Evaluation

| Term | Definition |
|---|---|
| **Golden Dataset** | Hand-written cases with known-correct expectations (`tests/eval/`), run by `make eval` against the real Groq model and real MCP servers. Unit tests use mocks and cannot see model-behavior or response-shape regressions. |
| **Eval vs Unit Test** | Unit tests prove the code calls the right things; evals prove the real model/servers produce correct output. Runs on PRs into `main`. |

## Memory & RAG

| Term | Definition |
|---|---|
| **Session Memory** | Conversation buffer for one ReAct run. A sliding window of the last 20 messages (message-count, not token-aware). Created fresh per run, never shared across requests. |
| **Patient Store** | Persistent per-patient memory in Postgres. Survives across sessions. Stores demographics, meds, labs, visits; every row carries `doctor_id`. Recalls only the most recent 5 visits. |
| **Visit** | A single doctor-patient encounter. Links to a session and records chief complaint, assessment, and plan. |
| **RAG** | Retrieval-Augmented Generation. Agent queries the vector store for relevant clinical guidelines before generating a response. |
| **Vector Store** | ChromaDB instance storing embedded chunks of clinical guidelines. Queried by semantic similarity. |
| **Chunk** | A segment of a clinical guideline document, typically 512 tokens with 64-token overlap. Stored with metadata (source, section, page). |
| **Embedding** | A dense vector representation of a text chunk, generated by sentence-transformers (`all-MiniLM-L6-v2`). Used for similarity search in ChromaDB. |
| **RAG Pipeline** | The full ingest flow: PDF → parse → chunk → embed → store in ChromaDB. Run once per guideline update. |

## Workflows

| Term | Definition |
|---|---|
| **Workflow** | A multi-agent orchestration pattern. Defines which agents run, in what order, with what data. |
| **Patient Assessment Workflow** | New patient / assess flow, a LangGraph graph: init_run (token tracker) → triage → parallel(drug_safety, evidence, trial_finder -- whichever triage selects) → report. Self-contained, so it runs standalone and embeds as a **subgraph** in the follow-up flow. |
| **Follow-Up Workflow** | Return visit flow: load the patient → run the assessment subgraph → (WebSocket only) pause at the approval node. The workflow never saves; the caller persists according to the `ApprovalOutcome`. |
| **Subgraph** | A compiled graph used as a node in another graph. The assessment graph is a node of the follow-up graph; with `subgraphs=True` streaming, its inner steps report progress. |
| **Checkpointer** | LangGraph's store for a paused run's state. Ours is in-memory only: a checkpoint holds the whole `PatientContext` and draft report (patient data), so it stays in process memory instead of a table outside the `doctor_id`-isolated schema. Loses drafts on restart; single replica only. |
| **Interrupt** | `interrupt()` pauses a graph at a node until the caller resumes it with `Command(resume=...)`. The node re-runs from its start on resume, so nothing before the interrupt may have side effects. |
| **Progress Event** | `{"type":"progress","step":...,"status":"completed\|failed\|skipped"}` streamed over the WebSocket for triage, each specialist, and the report. |
| **Drug Check Workflow** | Focused flow: enumerate all medication pairs → check each for interactions → aggregate risk scores. |

## Infrastructure

| Term | Definition |
|---|---|
| **Retry** | `@retry`: exponential backoff + jitter, applied to LLM and MCP calls. Retries only errors marked `retryable`, so a permanent failure (404, 400) is attempted once. The one place retries happen -- SDK-level retries are turned off. |
| **Retryable Error** | An `UpstreamError` with `retryable=True`: timeouts, connection failures, 5xx, and short rate limits. `client_error` (bad request, unknown model) is never retryable; an unclassified error defaults to not retryable. |
| **Retry-After** | The wait an upstream asks for on a rate limit (header, or Groq's "try again in 8m7s" message). Honoured if it is within `RETRY_MAX_WAIT_SECONDS`; otherwise the request fails fast rather than being held open for minutes. |
| **Retry Budget** | `RETRY_BUDGET_SECONDS`: the total time one call may spend retrying, so a run of slow failures cannot stack up. |
| **Rate Limiter** | Token-bucket rate limiter per API source. Prevents hitting external API rate limits (e.g., OpenFDA: 240 req/min). |
| **Circuit Breaker** | Guards each MCP server and the LLM. After 5 consecutive *upstream-health* failures (timeout, connection, 5xx) it opens and calls fail fast with the dependency's own error type (`circuit_open`) instead of hanging on a dead server; after 30s it lets a trial call through (half-open). 4xx and rate limits never count -- the server is up. State is per process. |
| **Structured Logging** | JSON logs via structlog. Every entry has an ISO UTC timestamp and level, plus whatever fields the call site passes (e.g. `run_id`, `step`, `source`) and any bound context vars. No field is added automatically. |
| **Tracing** | OpenTelemetry spans, one per HTTP request (middleware) and one per MCP HTTP call; optional LangSmith tracing of LLM calls (`LANGCHAIN_TRACING_V2`). Not a single end-to-end trace of the agent flow. |
| **Settings** | The `Settings` singleton from pydantic-settings. Reads from env vars → `.env` → defaults. Single source of truth for all config. |
