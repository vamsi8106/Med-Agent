from medagent.core.exceptions import (
    AuthError,
    MCPError,
    MedAgentError,
    MemoryError,
    PatientNotFoundError,
    ProviderError,
    ToolError,
)


def test_base_error_defaults_to_400() -> None:
    assert MedAgentError("boom").status_code == 400


def test_tool_error_maps_to_400() -> None:
    assert ToolError("bad input").status_code == 400


def test_provider_error_maps_to_502() -> None:
    assert ProviderError("llm down").status_code == 502


def test_mcp_error_maps_to_502() -> None:
    assert MCPError("mcp server down").status_code == 502


def test_memory_error_maps_to_500() -> None:
    assert MemoryError("db down").status_code == 500


def test_auth_error_maps_to_401() -> None:
    assert AuthError("bad token").status_code == 401


def test_patient_not_found_error_maps_to_404() -> None:
    assert PatientNotFoundError("unknown patient").status_code == 404


# --- rate limits are 429, not outages ------------------------------------------------------

import pytest  # noqa: E402

from medagent.core.exceptions import AllAgentsFailedError, UpstreamError, wait_phrase  # noqa: E402


@pytest.mark.parametrize("error_type", [ProviderError, MCPError])
def test_a_rate_limited_upstream_error_is_a_429(error_type: type[UpstreamError]) -> None:
    assert error_type("slow down", reason="rate_limited", retryable=True).status_code == 429


@pytest.mark.parametrize("reason", ["timeout", "server_error", "connection_error", "client_error"])
def test_every_other_upstream_reason_stays_a_502(reason: str) -> None:
    assert ProviderError("x", reason=reason).status_code == 502


def test_the_class_level_status_code_is_unchanged() -> None:
    assert ProviderError.status_code == 502
    assert MCPError.status_code == 502


def test_all_agents_failed_is_a_429_only_when_they_were_all_rate_limited() -> None:
    assert AllAgentsFailedError("x").status_code == 502
    assert AllAgentsFailedError("x", reason="rate_limited", retry_after=5).status_code == 429


def test_public_message_never_contains_the_raw_message() -> None:
    error = ProviderError("Error 429 org_01SECRET http://internal-host:3000", reason="server_error")
    assert error.public_message() == "the language model service was unavailable"
    assert "org_01SECRET" not in error.public_message()


def test_public_message_for_a_rate_limit_says_how_long_to_wait() -> None:
    error = MCPError("x", reason="rate_limited", retry_after=8.2)
    assert error.public_message() == (
        "an external medical data source is rate-limited; try again in about 9 seconds"
    )


@pytest.mark.parametrize(
    ("seconds", "phrase"),
    [
        (None, "shortly"),
        (0.2, "in about 1 seconds"),
        (8.2, "in about 9 seconds"),
        (89, "in about 89 seconds"),
        (90, "in about 2 minutes"),
        (487.7, "in about 9 minutes"),
    ],
)
def test_wait_phrase(seconds: float | None, phrase: str) -> None:
    assert wait_phrase(seconds) == phrase
