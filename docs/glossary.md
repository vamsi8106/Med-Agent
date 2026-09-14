# Glossary

Canonical names for every domain concept in MedAgent. Use these exact terms in code, docs, conversations, and commit messages. Update this file in the same PR that introduces or renames a term.

---

## Clinical Domain

| Term | Definition |
|---|---|
| **Patient Context** | The complete profile of a patient: demographics, active medications, lab results, conditions, allergies, visit history. Stored in SQLite, loaded into agent context per query. |
| **Medication** | A drug a patient is currently taking or has taken. Normalized to generic name via RxNorm. Tracks dose, frequency, route, status (active/discontinued/on_hold). |
| **Lab Result** | A single lab test value with units, reference range, and abnormal flag. Timestamped. Examples: HbA1c 8.9%, eGFR 52 mL/min/1.73m². |
| **Drug Interaction** | A clinically significant effect when two or more drugs are taken together. Graded by InteractionSeverity. Source-attributed (OpenFDA, medical-mcp). |
| **Evidence Grade** | Strength of clinical evidence supporting a recommendation. Enum: `high`, `moderate`, `low`, `very_low` (GRADE system). |
| **Interaction Severity** | Risk level of a drug-drug interaction. Enum: `low`, `moderate`, `high`, `critical`. |
| **CKD Stage** | Chronic kidney disease stage derived from eGFR. Enum: `stage_1` (≥90), `stage_2` (60-89), `stage_3a` (45-59), `stage_3b` (30-44), `stage_4` (15-29), `stage_5` (<15). |
| **Trial Phase** | Clinical trial phase. Enum: `phase_1`, `phase_2`, `phase_3`, `phase_4`, `not_applicable`. |
| **Clinical Evidence** | A piece of evidence retrieved from PubMed or RAG: source, title, summary, evidence grade, citation ID. |
| **Differential Diagnosis** | A ranked list of possible diagnoses given symptoms, history, and lab results. Not a definitive diagnosis. |
| **Chief Complaint** | The primary reason the patient is being seen. Free text recorded per visit. |
| **Reference Range** | Normal value boundaries for a lab test, adjusted by age, sex, and conditions. |

## System Architecture

| Term | Definition |
|---|---|
| **LLM Provider** | An abstraction over a specific LLM API (Groq, Anthropic, OpenAI). Implements `BaseLLMProvider`. Swappable via `LLM_PROVIDER` config. |
| **Provider Registry** | Singleton that maps provider names to `BaseLLMProvider` implementations. `get_provider("groq")` returns the Groq provider instance. |
| **Mock Provider** | A deterministic `BaseLLMProvider` for testing. Returns canned responses. No network. No API key. |
| **Tool** | A callable capability an agent can invoke. Implements `BaseTool`. Has a name, description, JSON schema for parameters, and an async `execute()` method. |
| **Tool Registry** | Singleton that tracks all available tools. Supports auto-discovery and `get_tool(name)`. |
| **@tool Decorator** | Decorator that converts a typed async function into a `BaseTool` with auto-generated JSON schema from type hints. |
| **MCP Client** | A client that reaches an MCP-adjacent server over HTTP (each server runs as its own container), calls its tools, and returns structured results. Only medical-mcp speaks the real MCP protocol under the hood, via a stdio<->HTTP bridge sidecar; healthcare-mcp and med-research-mcp-suite expose plain REST APIs. |
| **MCP Server** | A containerized Node.js process that exposes medical tools. MedAgent integrates three: medical-mcp, healthcare-mcp, med-research-mcp-suite -- see AGENTS.md's "MCP Server Config" for each one's real transport/contract. |

## Agent System

| Term | Definition |
|---|---|
| **Agent** | An autonomous unit that runs a perceive → think → act loop. Implements `BaseAgent`. Has access to an LLM provider, tools, and memory. |
| **Agent Role** | The specialization of an agent. Enum: `triage`, `drug_safety`, `evidence`, `trial_finder`, `report`. |
| **Triage Agent** | The routing agent. Analyzes the doctor's query and dispatches to one or more specialist agents. Does not call tools directly. |
| **Drug Safety Agent** | Checks drug interactions, adverse events, contraindications. Primary tools: `interaction_checker`, `medical-mcp/search-drugs`. |
| **Evidence Agent** | Searches literature and guidelines. Primary tools: `guideline_retriever`, `pubmedmcp/search-medical-literature`. |
| **Trial Finder Agent** | Searches ClinicalTrials.gov for matching trials. Primary tool: `healthcare-mcp/clinical_trials_search`. |
| **Report Agent** | Synthesizes outputs from other agents into a structured clinical report with citations. |
| **Agent Context** | The runtime context passed to an agent: patient context, conversation history, available tools, config. |
| **Agent Result** | The structured output of an agent run: response text, tool calls made, evidence cited, confidence level. |
| **ReAct Loop** | Reasoning + Acting pattern. Agent reasons about the task, selects a tool, observes the result, reasons again, until the task is complete or max steps reached. |

## Memory & RAG

| Term | Definition |
|---|---|
| **Session Memory** | Within-session conversation buffer. Sliding window, token-aware. Cleared when session ends. |
| **Patient Store** | Persistent per-patient memory in SQLite. Survives across sessions. Stores demographics, meds, labs, visits. |
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
| **Patient Assessment Workflow** | New patient flow: triage → parallel(drug_safety, evidence) → report. |
| **Follow-Up Workflow** | Return visit flow: recall patient memory → check changes → run relevant agents → update records. |
| **Drug Check Workflow** | Focused flow: enumerate all medication pairs → check each for interactions → aggregate risk scores. |

## Infrastructure

| Term | Definition |
|---|---|
| **Retry** | `@retry` decorator with exponential backoff + jitter. Applied to LLM calls and MCP tool calls. |
| **Rate Limiter** | Token-bucket rate limiter per API source. Prevents hitting external API rate limits (e.g., OpenFDA: 240 req/min). |
| **Circuit Breaker** | Monitors MCP server health. After N consecutive failures, opens the circuit and returns a fallback response. Closes after a cooldown period. |
| **Structured Logging** | JSON-formatted logs via structlog. Every log entry includes: timestamp, level, module, patient_id (when available), agent_role, tool_name. |
| **Tracing** | OpenTelemetry spans per patient query. Traces the full flow: query → triage → agent(s) → tools → response. |
| **Settings** | The `Settings` singleton from pydantic-settings. Reads from env vars → `.env` → defaults. Single source of truth for all config. |
