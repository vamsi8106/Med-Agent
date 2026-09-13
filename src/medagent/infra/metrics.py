"""Prometheus metrics for request and agent instrumentation."""

from prometheus_client import Counter, Histogram

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
    "Total agent invocations",
    labelnames=("agent_role", "outcome"),
)
