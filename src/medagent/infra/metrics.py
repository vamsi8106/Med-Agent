"""Prometheus metrics for request, agent and dependency instrumentation.

Every label has a small fixed set of values (never a patient id, query text, or
error message) so cardinality stays bounded.
"""

from prometheus_client import Counter, Gauge, Histogram

http_requests_total = Counter(
    "medagent_http_requests_total",
    "Total HTTP requests handled",
    labelnames=("method", "path", "status_code"),
)

http_request_duration_seconds = Histogram(
    "medagent_http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=("method", "path"),
)

agent_run_total = Counter(
    "medagent_agent_run_total",
    "Specialist agent steps by outcome (completed, failed, skipped)",
    labelnames=("agent_role", "outcome"),
)

llm_calls_total = Counter(
    "medagent_llm_calls_total",
    "LLM calls by final outcome (success, rate_limited, timeout, server_error, "
    "connection_error, client_error, circuit_open, error)",
    labelnames=("provider", "outcome"),
)

llm_call_duration_seconds = Histogram(
    "medagent_llm_call_duration_seconds",
    "LLM call duration in seconds, including retries",
    labelnames=("provider",),
    buckets=(0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120),
)

llm_tokens_total = Counter(
    "medagent_llm_tokens_total",
    "LLM tokens consumed",
    labelnames=("provider", "kind"),
)

retry_attempts_total = Counter(
    "medagent_retry_attempts_total",
    "Retries actually performed, by target (llm, mcp) and the reason for the retry",
    labelnames=("target", "reason"),
)

circuit_breaker_state = Gauge(
    "medagent_circuit_breaker_state",
    "Circuit breaker state as of its last transition (0 closed, 1 half-open, 2 open)",
    labelnames=("target",),
)

circuit_breaker_opens_total = Counter(
    "medagent_circuit_breaker_opens_total",
    "Times a circuit breaker opened",
    labelnames=("target",),
)

mcp_requests_total = Counter(
    "medagent_mcp_requests_total",
    "MCP server requests by final outcome",
    labelnames=("server", "outcome"),
)

tool_calls_total = Counter(
    "medagent_tool_calls_total",
    "ReAct tool calls by outcome (success, error, duplicate)",
    labelnames=("tool", "outcome"),
)

evidence_react_total = Counter(
    "medagent_evidence_react_total",
    "Evidence agent runs by path (completed, fallback_ungrounded, fallback_timeout, "
    "fallback_error)",
    labelnames=("outcome",),
)

guardrail_flags_total = Counter(
    "medagent_guardrail_flags_total",
    "Guardrail findings on LLM output (allergy_mention, output_validation, unverified_figures)",
    labelnames=("kind",),
)

context_truncations_total = Counter(
    "medagent_context_truncations_total",
    "Times text was cut to fit a context budget",
    labelnames=("source",),
)
