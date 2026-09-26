from medagent.api.errors import doctor_scope, error_payload, retry_after
from medagent.core.exceptions import MedAgentError, PatientNotFoundError, ProviderError
from medagent.core.models import User


def _user(role: str) -> User:
    return User(id="U-TEST-001", username="dr.alpha", role=role)


def test_upstream_error_text_is_replaced_by_fixed_public_message() -> None:
    exc = ProviderError("boom at http://internal-host:9000 account acct-123", reason="server_error")

    detail = error_payload(exc)["detail"]

    assert "internal-host" not in detail
    assert "acct-123" not in detail


def test_own_error_text_is_passed_through() -> None:
    assert error_payload(PatientNotFoundError("Unknown patient: P-TEST-001"))["detail"] == (
        "Unknown patient: P-TEST-001"
    )


def test_retry_after_only_advertised_for_a_429_that_knows_it() -> None:
    limited = ProviderError("slow down", reason="rate_limited", retry_after=12.0)
    server = ProviderError("down", reason="server_error", retry_after=12.0)

    assert retry_after(limited) == 12.0
    assert error_payload(limited)["retry_after"] == 12.0
    assert retry_after(server) is None
    assert "retry_after" not in error_payload(server)
    assert retry_after(MedAgentError("plain")) is None


def test_admin_is_unscoped_and_a_doctor_is_scoped_to_themselves() -> None:
    assert doctor_scope(_user("admin")) is None
    assert doctor_scope(_user("doctor")) == "U-TEST-001"
