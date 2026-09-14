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
