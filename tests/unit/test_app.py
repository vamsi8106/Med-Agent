import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from medagent.app import create_app
from medagent.auth.security import verify_password
from medagent.core.config import Settings
from medagent.core.exceptions import (
    AllAgentsFailedError,
    MCPError,
    PatientNotFoundError,
    ProviderError,
    ToolError,
)
from medagent.core.models import AgentResult, PatientContext
from medagent.core.types import AgentRole
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore

_ADMIN_USERNAME = "admin"
_ADMIN_PASSWORD = "admin-pass-1234"
_PASSWORD = "s3cret-password-1"


class _FakeEmbeddingModel:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]


class _FakeEvidenceAgent:
    """Stands in for the LLM-backed agent so the real graph, checkpointer and
    WebSocket handler run end to end with no network."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def gather_evidence(self, context: PatientContext, message: str) -> AgentResult:
        return AgentResult(role=AgentRole.EVIDENCE, summary="Follow-up evidence summary")


class _FakeVectorStore:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def query(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
        return []

    async def ping(self) -> bool:
        return True


def _settings(pg_dsn: str, **overrides: Any) -> Settings:
    # Abuse limits are generous by default so tests that log in or assess many
    # times are not throttled; the limit tests pass their own values.
    fields: dict[str, Any] = {
        "llm_provider": "mock",
        "postgres_dsn": pg_dsn,
        "admin_bootstrap_username": _ADMIN_USERNAME,
        "admin_bootstrap_password": _ADMIN_PASSWORD,
        "login_attempts_burst": 1000,
        "login_attempts_per_ip_burst": 1000,
        "agent_runs_burst": 1000,
    }
    fields.update(overrides)
    return Settings(_env_file=None, **fields)


@contextmanager
def _make_client(pg_dsn: str, **overrides: Any) -> Iterator[TestClient]:
    # AppState (and therefore VectorStore/EmbeddingModel construction) only
    # happens inside FastAPI's lifespan, which TestClient triggers on
    # __enter__ -- so these patches must still be active at that point, not
    # just while create_app() itself runs.
    with (
        patch("medagent.api.state.EmbeddingModel", _FakeEmbeddingModel),
        patch("medagent.api.state.VectorStore", _FakeVectorStore),
        patch("medagent.api.state.EvidenceAgent", _FakeEvidenceAgent),
        TestClient(create_app(_settings(pg_dsn, **overrides))) as client,
    ):
        yield client


def _login(client: TestClient, username: str, password: str) -> str:
    response = client.post("/auth/token", data={"username": username, "password": password})
    return response.json()["access_token"]


def _admin_token(client: TestClient) -> str:
    return _login(client, _ADMIN_USERNAME, _ADMIN_PASSWORD)


def _admin_headers(client: TestClient) -> dict[str, str]:
    return {"Authorization": f"Bearer {_admin_token(client)}"}


def _auth_headers(client: TestClient, username: str = "dr.alpha") -> dict[str, str]:
    """Log in as the bootstrapped admin, create a doctor account, log in as them."""
    client.post(
        "/admin/users",
        json={"username": username, "password": _PASSWORD},
        headers=_admin_headers(client),
    )
    token = _login(client, username, _PASSWORD)
    return {"Authorization": f"Bearer {token}"}


def test_health_endpoint(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert b"medagent_http_requests_total" in response.content


def test_admin_is_bootstrapped_on_startup(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        response = client.post(
            "/auth/token", data={"username": _ADMIN_USERNAME, "password": _ADMIN_PASSWORD}
        )
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"


def test_admin_endpoint_requires_admin_role(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        doctor_headers = _auth_headers(client, username="dr.notadmin")
        response = client.post(
            "/admin/users",
            json={"username": "dr.other", "password": _PASSWORD},
            headers=doctor_headers,
        )
    assert response.status_code == 403


def test_admin_endpoint_rejects_unauthenticated(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        response = client.post("/admin/users", json={"username": "dr.new", "password": _PASSWORD})
    assert response.status_code == 401


def test_admin_creates_user_and_they_can_log_in(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        create_resp = client.post(
            "/admin/users",
            json={"username": "dr.beta", "password": _PASSWORD},
            headers=_admin_headers(client),
        )
        assert create_resp.status_code == 200
        assert create_resp.json()["username"] == "dr.beta"
        assert create_resp.json()["role"] == "doctor"

        login_resp = client.post("/auth/token", data={"username": "dr.beta", "password": _PASSWORD})
    assert login_resp.status_code == 200


def test_login_with_wrong_password_returns_401(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        client.post(
            "/admin/users",
            json={"username": "dr.gamma", "password": _PASSWORD},
            headers=_admin_headers(client),
        )
        response = client.post("/auth/token", data={"username": "dr.gamma", "password": "wrong"})
    assert response.status_code == 401


def test_duplicate_username_returns_409(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        headers = _admin_headers(client)
        client.post(
            "/admin/users",
            json={"username": "dr.delta", "password": _PASSWORD},
            headers=headers,
        )
        response = client.post(
            "/admin/users",
            json={"username": "dr.delta", "password": "other-password-2"},
            headers=headers,
        )
    assert response.status_code == 409


def test_patient_endpoints_require_auth(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        response = client.get("/patients/P-TEST-800")
    assert response.status_code == 401


def test_create_and_get_patient_with_auth(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client)
        payload = {"id": "P-TEST-800", "name": "Patient Alpha", "age": 55, "sex": "F"}
        create_resp = client.post("/patients", json=payload, headers=headers)
        assert create_resp.status_code == 200

        get_resp = client.get("/patients/P-TEST-800", headers=headers)

    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Patient Alpha"


def test_get_missing_patient_returns_404(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client)
        response = client.get("/patients/P-MISSING", headers=headers)
    assert response.status_code == 404


def test_get_patient_from_another_doctor_returns_404(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        owner_headers = _auth_headers(client, username="dr.owner")
        client.post(
            "/patients",
            json={"id": "P-TEST-807", "name": "Patient Eta", "age": 44, "sex": "F"},
            headers=owner_headers,
        )

        other_headers = _auth_headers(client, username="dr.other")
        response = client.get("/patients/P-TEST-807", headers=other_headers)

    assert response.status_code == 404


def test_admin_can_view_any_doctors_patient(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        owner_headers = _auth_headers(client, username="dr.owner2")
        client.post(
            "/patients",
            json={"id": "P-TEST-808", "name": "Patient Theta", "age": 51, "sex": "M"},
            headers=owner_headers,
        )

        response = client.get("/patients/P-TEST-808", headers=_admin_headers(client))

    assert response.status_code == 200
    assert response.json()["name"] == "Patient Theta"


def test_assess_endpoint_scoped_to_owning_doctor(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        owner_headers = _auth_headers(client, username="dr.owner3")
        client.post(
            "/patients",
            json={"id": "P-TEST-809", "name": "Patient Iota", "age": 39, "sex": "F"},
            headers=owner_headers,
        )

        other_headers = _auth_headers(client, username="dr.other3")
        response = client.post(
            "/patients/P-TEST-809/assess", json={"message": "hi"}, headers=other_headers
        )

    assert response.status_code == 404


def test_drug_check_endpoint_scoped_to_owning_doctor(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        owner_headers = _auth_headers(client, username="dr.owner4")
        client.post(
            "/patients",
            json={"id": "P-TEST-810", "name": "Patient Kappa", "age": 62, "sex": "M"},
            headers=owner_headers,
        )

        other_headers = _auth_headers(client, username="dr.other4")
        response = client.post(
            "/drug-check",
            json={"patient_id": "P-TEST-810", "new_drug": "Glimepiride"},
            headers=other_headers,
        )

    assert response.status_code == 404


def test_audit_log_records_create_and_view(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client, username="dr.audit")
        client.post(
            "/patients",
            json={"id": "P-TEST-806", "name": "Patient Zeta", "age": 33, "sex": "F"},
            headers=headers,
        )
        client.get("/patients/P-TEST-806", headers=headers)

        response = client.get("/patients/P-TEST-806/audit-log", headers=headers)

    assert response.status_code == 200
    actions = [entry["action"] for entry in response.json()]
    # newest first; the audit_log_viewed entry for this very request is included too
    assert "patient_viewed" in actions
    assert "patient_created" in actions
    assert all(entry["username"] == "dr.audit" for entry in response.json())


def test_assess_endpoint_returns_report(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client)
        client.post(
            "/patients",
            json={"id": "P-TEST-801", "name": "Patient Beta", "age": 60, "sex": "M"},
            headers=headers,
        )
        with patch(
            "medagent.api.routes.patients.run_patient_assessment",
            AsyncMock(return_value="# Report"),
        ):
            response = client.post(
                "/patients/P-TEST-801/assess", json={"message": "hi"}, headers=headers
            )

    assert response.status_code == 200
    assert response.json() == {"report": "# Report"}


def test_assess_endpoint_creates_visit_record(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client, username="dr.visit1")
        client.post(
            "/patients",
            json={"id": "P-TEST-820", "name": "Patient Omega", "age": 60, "sex": "M"},
            headers=headers,
        )
        with patch(
            "medagent.api.routes.patients.run_patient_assessment",
            AsyncMock(return_value="# Report"),
        ):
            client.post(
                "/patients/P-TEST-820/assess", json={"message": "headache"}, headers=headers
            )

    store = PatientStore(PersistentStore(pg_dsn))
    loaded = asyncio.run(store.get_patient("P-TEST-820"))
    assert loaded is not None
    assert len(loaded.visits) == 1
    assert loaded.visits[0].chief_complaint == "headache"
    assert loaded.visits[0].assessment == "# Report"


def test_drug_check_endpoint(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client)
        with patch(
            "medagent.api.routes.patients.run_drug_check",
            AsyncMock(return_value="No major concerns."),
        ):
            response = client.post(
                "/drug-check",
                json={"patient_id": "P-TEST-802", "new_drug": "Glimepiride"},
                headers=headers,
            )

    assert response.status_code == 200
    assert response.json() == {"answer": "No major concerns."}


def test_mcp_error_maps_to_502(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client)
        with patch(
            "medagent.api.routes.patients.run_drug_check",
            AsyncMock(side_effect=MCPError("upstream down")),
        ):
            response = client.post(
                "/drug-check",
                json={"patient_id": "P-TEST-803", "new_drug": "Glimepiride"},
                headers=headers,
            )

    assert response.status_code == 502
    # Fixed text, not the raw message: MCP errors carry internal URLs/hostnames.
    assert response.json() == {"detail": "an external medical data source was unavailable"}
    assert "retry-after" not in response.headers


def test_patient_not_found_error_maps_to_404(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client)
        with patch(
            "medagent.api.routes.patients.run_drug_check",
            AsyncMock(side_effect=PatientNotFoundError("no such patient")),
        ):
            response = client.post(
                "/drug-check",
                json={"patient_id": "P-TEST-804", "new_drug": "Glimepiride"},
                headers=headers,
            )

    assert response.status_code == 404
    assert response.json() == {"detail": "no such patient"}


def _assert_closed_for_policy(websocket: WebSocketTestSession) -> None:
    with pytest.raises(WebSocketDisconnect) as closed:
        websocket.receive_json()
    assert closed.value.code == 1008


@pytest.mark.parametrize(
    "first_message",
    [
        "any updates?",  # not an auth frame
        json.dumps({"type": "auth"}),  # no token
        json.dumps({"type": "auth", "token": "not-a-jwt"}),
    ],
)
def test_websocket_without_valid_auth_message_is_closed(pg_dsn: str, first_message: str) -> None:
    with _make_client(pg_dsn) as client, client.websocket_connect("/ws/P-TEST-999") as websocket:
        websocket.send_text(first_message)
        _assert_closed_for_policy(websocket)


def test_websocket_token_in_url_is_not_accepted(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        token = _admin_token(client)
        with client.websocket_connect(f"/ws/P-TEST-999?token={token}") as websocket:
            websocket.send_text("any updates?")
            _assert_closed_for_policy(websocket)


def test_websocket_rejects_oversized_message(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        token = _admin_token(client)
        _create_patient(client, token, "P-TEST-809", "Patient Iota")
        with _ws(client, "P-TEST-809", token) as websocket:
            websocket.send_text("x" * 4001)
            frame = websocket.receive_json()
            assert frame["type"] == "error"
            assert "1-4000" in frame["detail"]


def _create_patient(client: TestClient, token: str, patient_id: str, name: str) -> None:
    response = client.post(
        "/patients",
        json={"id": patient_id, "name": name, "age": 70, "sex": "M"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


@contextmanager
def _ws(
    client: TestClient, patient_id: str, token: str, resume: bool = False
) -> Iterator[WebSocketTestSession]:
    """Connects and authenticates with the first message, as real clients must."""
    url = f"/ws/{patient_id}" + ("?resume=1" if resume else "")
    with client.websocket_connect(url) as websocket:
        websocket.send_json({"type": "auth", "token": token})
        yield websocket


def _draft(websocket: WebSocketTestSession, message: str) -> tuple[list[dict], dict]:
    """Sends a message and returns (progress frames, the pending_approval frame)."""
    websocket.send_text(message)
    progress = []
    while True:
        frame = websocket.receive_json()
        if frame["type"] != "progress":
            return progress, frame
        progress.append(frame)


def _visits(pg_dsn: str, patient_id: str) -> list:
    store = PatientStore(PersistentStore(pg_dsn))
    loaded = asyncio.run(store.get_patient(patient_id))
    assert loaded is not None
    return loaded.visits


def test_websocket_chat_streams_progress_then_requires_approval_before_saving(
    pg_dsn: str,
) -> None:
    with _make_client(pg_dsn) as client:
        token = _admin_token(client)
        _create_patient(client, token, "P-TEST-803", "Patient Gamma")
        with _ws(client, "P-TEST-803", token) as websocket:
            progress, pending = _draft(websocket, "any updates?")
            assert [(f["step"], f["status"]) for f in progress] == [
                ("triage", "completed"),
                ("evidence", "completed"),
                ("report", "completed"),
            ]
            assert pending["type"] == "pending_approval"
            assert "Follow-up evidence summary" in pending["report"]
            assert "resumed" not in pending
            assert _visits(pg_dsn, "P-TEST-803") == []

            websocket.send_text("approve")
            saved = websocket.receive_json()
            assert saved == {"type": "saved", "report": pending["report"]}

    visits = _visits(pg_dsn, "P-TEST-803")
    assert len(visits) == 1
    assert visits[0].chief_complaint == "any updates?"


def test_websocket_chat_reject_does_not_save(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        token = _admin_token(client)
        _create_patient(client, token, "P-TEST-804", "Patient Delta")
        with _ws(client, "P-TEST-804", token) as websocket:
            _draft(websocket, "any updates?")
            websocket.send_text("reject")
            assert websocket.receive_json() == {"type": "rejected"}

    assert _visits(pg_dsn, "P-TEST-804") == []


def test_websocket_chat_edited_text_is_saved_instead(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        token = _admin_token(client)
        _create_patient(client, token, "P-TEST-805", "Patient Epsilon")
        with _ws(client, "P-TEST-805", token) as websocket:
            _draft(websocket, "any updates?")
            websocket.send_text("Edited: monitor renal function closely.")
            saved = websocket.receive_json()
            assert saved == {"type": "saved", "report": "Edited: monitor renal function closely."}

    visits = _visits(pg_dsn, "P-TEST-805")
    assert [v.assessment for v in visits] == ["Edited: monitor renal function closely."]


def test_websocket_draft_survives_a_dropped_connection_and_resumes(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        token = _admin_token(client)
        _create_patient(client, token, "P-TEST-806", "Patient Zeta")
        with _ws(client, "P-TEST-806", token) as websocket:
            _, pending = _draft(websocket, "how is she doing?")
        # connection dropped with the draft still awaiting a decision
        assert _visits(pg_dsn, "P-TEST-806") == []

        with _ws(client, "P-TEST-806", token, resume=True) as websocket:
            resumed = websocket.receive_json()
            assert resumed == {
                "type": "pending_approval",
                "report": pending["report"],
                "resumed": True,
            }
            websocket.send_text("approve")
            assert websocket.receive_json()["type"] == "saved"

    visits = _visits(pg_dsn, "P-TEST-806")
    assert len(visits) == 1
    # the visit records the question asked before the drop, not the reconnect
    assert visits[0].chief_complaint == "how is she doing?"


def test_reconnecting_without_resume_discards_the_draft_so_a_new_question_is_not_an_edit(
    pg_dsn: str,
) -> None:
    with _make_client(pg_dsn) as client:
        token = _admin_token(client)
        _create_patient(client, token, "P-TEST-807", "Patient Eta")
        with _ws(client, "P-TEST-807", token) as websocket:
            _draft(websocket, "first question")

        with _ws(client, "P-TEST-807", token) as websocket:
            # Without resume=1 this is a fresh question, NOT the reply to the
            # stale draft -- it must not be saved as an "edited" report.
            _, pending = _draft(websocket, "a brand new question")
            assert pending["type"] == "pending_approval"
            websocket.send_text("reject")
            assert websocket.receive_json() == {"type": "rejected"}

    assert _visits(pg_dsn, "P-TEST-807") == []


def test_another_user_cannot_resume_someone_elses_draft(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        admin_token = _admin_token(client)
        _create_patient(client, admin_token, "P-TEST-808", "Patient Theta")
        with _ws(client, "P-TEST-808", admin_token) as websocket:
            _draft(websocket, "any updates?")

        headers = _auth_headers(client, "dr.intruder")
        intruder_token = headers["Authorization"].removeprefix("Bearer ")
        with _ws(client, "P-TEST-808", intruder_token, resume=True) as websocket:
            websocket.send_text("any updates?")
            frame = websocket.receive_json()
            # no draft handed over, and the patient isn't theirs to draft for
            assert frame["type"] == "error"
            assert "P-TEST-808" in frame["detail"]

    assert _visits(pg_dsn, "P-TEST-808") == []


# --- rate limits over HTTP and WebSocket ---------------------------------------------------


def test_a_rate_limited_assessment_is_a_429_with_retry_after(pg_dsn: str) -> None:
    error = AllAgentsFailedError("all throttled", reason="rate_limited", retry_after=7.2)
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client)
        client.post(
            "/patients",
            json={"id": "P-TEST-820", "name": "Patient Iota", "age": 50, "sex": "F"},
            headers=headers,
        )
        with patch(
            "medagent.api.routes.patients.run_patient_assessment", AsyncMock(side_effect=error)
        ):
            response = client.post(
                "/patients/P-TEST-820/assess", json={"message": "hi"}, headers=headers
            )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "8"  # 7.2s rounded up
    assert response.json()["retry_after"] == 7.2


def test_a_rate_limited_upstream_error_on_drug_check_is_a_429_and_hides_the_raw_message(
    pg_dsn: str,
) -> None:
    error = ProviderError(
        "Groq 429 in organization org_01SECRET", reason="rate_limited", retry_after=30
    )
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client)
        with patch("medagent.api.routes.patients.run_drug_check", AsyncMock(side_effect=error)):
            response = client.post(
                "/drug-check",
                json={"patient_id": "P-TEST-803", "new_drug": "Glimepiride"},
                headers=headers,
            )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"
    assert response.json()["detail"] == (
        "the language model service is rate-limited; try again in about 30 seconds"
    )
    assert "org_01SECRET" not in response.text


def test_a_rate_limit_without_a_known_wait_has_no_retry_after_header(pg_dsn: str) -> None:
    error = ProviderError("x", reason="rate_limited")
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client)
        with patch("medagent.api.routes.patients.run_drug_check", AsyncMock(side_effect=error)):
            response = client.post(
                "/drug-check",
                json={"patient_id": "P-TEST-803", "new_drug": "Glimepiride"},
                headers=headers,
            )

    assert response.status_code == 429
    assert "retry-after" not in response.headers
    assert "retry_after" not in response.json()


def test_our_own_errors_keep_their_own_message(pg_dsn: str) -> None:
    """Only upstream errors are replaced by fixed text; a bad-input error is
    already written for the client."""
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client)
        with patch(
            "medagent.api.routes.patients.run_drug_check",
            AsyncMock(side_effect=ToolError("Need two medications")),
        ):
            response = client.post(
                "/drug-check",
                json={"patient_id": "P-TEST-803", "new_drug": "X"},
                headers=headers,
            )

    assert response.status_code == 400
    assert response.json() == {"detail": "Need two medications"}


class _RateLimitedEvidenceAgent(_FakeEvidenceAgent):
    async def gather_evidence(self, context: PatientContext, message: str) -> AgentResult:
        raise ProviderError("Groq 429 org_01SECRET", reason="rate_limited", retry_after=4.5)


def test_the_websocket_error_frame_carries_retry_after(pg_dsn: str) -> None:
    # the agent is built in the app's lifespan, so patch before the client starts
    with (
        patch("medagent.api.state.EmbeddingModel", _FakeEmbeddingModel),
        patch("medagent.api.state.VectorStore", _FakeVectorStore),
        patch("medagent.api.state.EvidenceAgent", _RateLimitedEvidenceAgent),
        TestClient(create_app(_settings(pg_dsn))) as client,
    ):
        token = _admin_token(client)
        _create_patient(client, token, "P-TEST-822", "Patient Lambda")
        with _ws(client, "P-TEST-822", token) as websocket:
            websocket.send_text("any updates?")
            while (frame := websocket.receive_json())["type"] == "progress":
                pass

    assert frame["type"] == "error"
    assert frame["retry_after"] == 4.5
    assert "rate-limited" in frame["detail"]
    assert "org_01SECRET" not in frame["detail"]


# --- /ready ------------------------------------------------------------------------------


def _all_mcp_probes(healthy: bool):
    from medagent.tools.mcp.http_base import HttpMCPClient

    return patch.object(HttpMCPClient, "probe", AsyncMock(return_value=healthy))


def test_ready_is_ok_when_every_dependency_answers(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client, _all_mcp_probes(True):
        response = client.get("/ready")  # unauthenticated, like /health

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["checks"]["postgres"] == "ok"


def test_ready_degrades_but_stays_200_when_an_mcp_server_is_down(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client, _all_mcp_probes(False):
        response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["postgres"] == "ok"
    assert body["checks"]["medical_mcp"] == "down"


def test_ready_is_503_when_postgres_is_down_and_reveals_no_detail(pg_dsn: str) -> None:
    from medagent.memory.persistent import PersistentStore

    boom = AsyncMock(side_effect=RuntimeError("connection to postgres://user:pw@db:5432 refused"))
    with (
        _make_client(pg_dsn) as client,
        _all_mcp_probes(True),
        patch.object(PersistentStore, "ping", boom),
    ):
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "down"
    assert "postgres://" not in response.text
    assert "refused" not in response.text


def test_health_stays_a_cheap_liveness_check_that_touches_nothing(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client, _all_mcp_probes(False):
        assert client.get("/health").json() == {"status": "ok"}


def test_posting_another_doctors_patient_id_is_refused_and_changes_nothing(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        owner = _auth_headers(client, "dr.alpha")
        intruder = _auth_headers(client, "dr.intruder")
        client.post(
            "/patients",
            json={
                "id": "P-TEST-OWN",
                "name": "Patient Alpha",
                "age": 70,
                "sex": "M",
                "medications": [{"name": "Metformin"}],
            },
            headers=owner,
        )

        response = client.post(
            "/patients",
            json={
                "id": "P-TEST-OWN",
                "name": "Overwritten",
                "age": 1,
                "sex": "F",
                "medications": [{"name": "Warfarin"}],
            },
            headers=intruder,
        )

        assert response.status_code == 409
        assert response.json() == {"detail": "This patient ID cannot be used."}
        record = client.get("/patients/P-TEST-OWN", headers=owner).json()
        assert record["name"] == "Patient Alpha"
        assert [m["name"] for m in record["medications"]] == ["Metformin"]


# --- abuse limits, input bounds, headers (security hardening) ---


def test_login_is_limited_per_username_even_with_the_right_password(pg_dsn: str) -> None:
    with _make_client(pg_dsn, login_attempts_burst=3, login_attempts_per_minute=1) as client:
        statuses = [
            client.post(
                "/auth/token", data={"username": _ADMIN_USERNAME, "password": "wrong-guess-123"}
            ).status_code
            for _ in range(3)
        ]
        limited = client.post(
            "/auth/token", data={"username": _ADMIN_USERNAME, "password": _ADMIN_PASSWORD}
        )
        other_user = client.post(
            "/auth/token", data={"username": "dr.unknown", "password": "whatever-123"}
        )

    assert statuses == [401, 401, 401]
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1
    assert other_user.status_code == 401  # a different account is not locked out


def test_login_is_limited_per_client_ip(pg_dsn: str) -> None:
    with _make_client(pg_dsn, login_attempts_per_ip_burst=2) as client:
        codes = [
            client.post(
                "/auth/token", data={"username": f"dr.user{i}", "password": "wrong-guess-123"}
            ).status_code
            for i in range(3)
        ]
    assert codes == [401, 401, 429]


def test_unknown_username_still_checks_a_password_hash(pg_dsn: str) -> None:
    with (
        _make_client(pg_dsn) as client,
        patch("medagent.api.routes.auth.verify_password", wraps=verify_password) as spy,
    ):
        response = client.post(
            "/auth/token", data={"username": "dr.nobody", "password": "whatever-123"}
        )
    assert response.status_code == 401
    spy.assert_called_once()


def test_agent_runs_are_limited_per_user(pg_dsn: str) -> None:
    with _make_client(pg_dsn, agent_runs_burst=2, agent_runs_per_minute=1) as client:
        alice = _auth_headers(client, "dr.alice")
        bob = _auth_headers(client, "dr.bob")
        _create_patient(client, alice["Authorization"].removeprefix("Bearer "), "P-TEST-950", "A")
        with patch("medagent.api.routes.patients.run_drug_check", AsyncMock(return_value="ok")):
            body = {"patient_id": "P-TEST-950", "new_drug": "Ibuprofen"}
            codes = [
                client.post("/drug-check", json=body, headers=alice).status_code for _ in range(3)
            ]
            bob_code = client.post("/drug-check", json=body, headers=bob).status_code

    assert codes == [200, 200, 429]
    assert bob_code != 429  # (bob gets 404 or similar: not his patient, but not throttled)


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/patients/P-TEST-951/assess", {"message": "x" * 4001}),
        ("/patients/P-TEST-951/assess", {"message": ""}),
        ("/patients", {"id": "P TEST/../1", "name": "Patient Alpha", "age": 70, "sex": "M"}),
        ("/patients", {"id": "P-TEST-952", "name": "Patient Alpha", "age": 400, "sex": "M"}),
        (
            "/patients",
            {
                "id": "P-TEST-953",
                "name": "Patient Alpha",
                "age": 70,
                "sex": "M",
                "medications": [{"name": f"Drug{i}"} for i in range(101)],
            },
        ),
        ("/drug-check", {"patient_id": "P-TEST-951", "new_drug": "x" * 201}),
    ],
)
def test_out_of_bounds_input_is_rejected(pg_dsn: str, path: str, body: dict) -> None:
    with _make_client(pg_dsn) as client:
        response = client.post(path, json=body, headers=_admin_headers(client))
    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"username": "dr.short", "password": "short"},
        {"username": "dr.role", "password": _PASSWORD, "role": "Admin"},
        {"username": "bad name!", "password": _PASSWORD},
    ],
)
def test_account_creation_validates_password_role_and_username(pg_dsn: str, body: dict) -> None:
    with _make_client(pg_dsn) as client:
        response = client.post("/admin/users", json=body, headers=_admin_headers(client))
    assert response.status_code == 422


def test_oversized_body_is_refused_before_parsing(pg_dsn: str) -> None:
    with _make_client(pg_dsn, max_request_body_bytes=1000) as client:
        response = client.post(
            "/patients",
            content=b"{" + b" " * 2000 + b"}",
            headers={**_admin_headers(client), "Content-Type": "application/json"},
        )
    assert response.status_code == 413
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_oversized_streamed_body_without_content_length_is_refused(pg_dsn: str) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(5):
            yield b" " * 500

    with _make_client(pg_dsn, max_request_body_bytes=1000) as client:
        response = client.post(
            "/patients",
            content=chunks(),
            headers={**_admin_headers(client), "Content-Type": "application/json"},
        )
    assert response.status_code == 413


def test_security_headers_on_every_response(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        response = client.get("/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"
    assert "Strict-Transport-Security" not in response.headers
    assert client.get("/docs").status_code == 200  # development keeps the docs


def test_production_hides_docs_and_adds_hsts(pg_dsn: str) -> None:
    with _make_client(pg_dsn, app_env="production", jwt_secret_key="x" * 40) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        response = client.get("/health")
    assert response.headers["Strict-Transport-Security"].startswith("max-age=")


def test_audit_log_honours_limit_and_rejects_huge_pages(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        headers = _admin_headers(client)
        _create_patient(client, headers["Authorization"].removeprefix("Bearer "), "P-TEST-954", "A")
        for _ in range(4):
            client.get("/patients/P-TEST-954", headers=headers)

        page = client.get("/patients/P-TEST-954/audit-log?limit=2", headers=headers)
        too_big = client.get("/patients/P-TEST-954/audit-log?limit=501", headers=headers)

    assert len(page.json()) == 2
    assert too_big.status_code == 422


# --- user management ---


def _bearer(headers: dict[str, str]) -> str:
    return headers["Authorization"].removeprefix("Bearer ")


def _audit_actions(pg_dsn: str) -> list[str]:
    async def fetch() -> list[str]:
        store = PersistentStore(pg_dsn)
        async with store.connect() as conn:
            rows = await conn.fetch("SELECT action FROM audit_log WHERE patient_id IS NULL")
        await store.close()
        return [row["action"] for row in rows]

    return asyncio.run(fetch())


def test_admin_lists_users_without_password_hashes(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        _auth_headers(client, "dr.listed")
        response = client.get("/admin/users", headers=_admin_headers(client))
    assert response.status_code == 200
    users = {u["username"]: u for u in response.json()}
    assert users["dr.listed"]["disabled"] is False
    assert all(
        set(u) == {"id", "username", "role", "disabled", "created_at"} for u in users.values()
    )


def test_disabling_a_user_blocks_their_token_and_login(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        doctor = _auth_headers(client, "dr.leaving")
        admin = _admin_headers(client)

        assert client.post("/admin/users/dr.leaving/disable", headers=admin).status_code == 204
        token_after = client.get("/patients/P-TEST-999", headers=doctor)
        login_after = client.post(
            "/auth/token", data={"username": "dr.leaving", "password": _PASSWORD}
        )

        assert client.post("/admin/users/dr.leaving/enable", headers=admin).status_code == 204
        login_enabled = client.post(
            "/auth/token", data={"username": "dr.leaving", "password": _PASSWORD}
        )

    assert token_after.status_code == 401
    assert login_after.status_code == 401
    assert login_after.json()["detail"] == "Incorrect username or password"  # same as wrong pw
    assert login_enabled.status_code == 200
    assert "user_disabled" in _audit_actions(pg_dsn)


def test_admin_cannot_disable_themselves(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        response = client.post(
            f"/admin/users/{_ADMIN_USERNAME}/disable", headers=_admin_headers(client)
        )
    assert response.status_code == 409


def test_reset_password_revokes_old_tokens(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        doctor = _auth_headers(client, "dr.forgot")
        response = client.post(
            "/admin/users/dr.forgot/reset-password",
            json={"new_password": "brand-new-password-1"},
            headers=_admin_headers(client),
        )
        old_token = client.get("/patients/P-TEST-999", headers=doctor)
        old_password = client.post(
            "/auth/token", data={"username": "dr.forgot", "password": _PASSWORD}
        )
        new_password = client.post(
            "/auth/token", data={"username": "dr.forgot", "password": "brand-new-password-1"}
        )
    assert response.status_code == 204
    assert old_token.status_code == 401
    assert old_password.status_code == 401
    assert new_password.status_code == 200


def test_revoke_tokens_ends_sessions_but_keeps_the_password(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        doctor = _auth_headers(client, "dr.lostphone")
        assert (
            client.post(
                "/admin/users/dr.lostphone/revoke-tokens", headers=_admin_headers(client)
            ).status_code
            == 204
        )
        old_token = client.get("/patients/P-TEST-999", headers=doctor)
        relogin = client.post(
            "/auth/token", data={"username": "dr.lostphone", "password": _PASSWORD}
        )
        fresh = client.get(
            "/patients/P-TEST-999",
            headers={"Authorization": f"Bearer {relogin.json()['access_token']}"},
        )
    assert old_token.status_code == 401
    assert fresh.status_code == 404  # authenticated again; the patient just doesn't exist


def test_change_password(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        doctor = _auth_headers(client, "dr.rotate")
        wrong = client.post(
            "/auth/change-password",
            json={"current_password": "not-my-password", "new_password": "rotated-password-1"},
            headers=doctor,
        )
        ok = client.post(
            "/auth/change-password",
            json={"current_password": _PASSWORD, "new_password": "rotated-password-1"},
            headers=doctor,
        )
        old_token = client.get("/patients/P-TEST-999", headers=doctor)
        relogin = client.post(
            "/auth/token", data={"username": "dr.rotate", "password": "rotated-password-1"}
        )
    assert wrong.status_code == 401
    assert ok.status_code == 204
    assert old_token.status_code == 401
    assert relogin.status_code == 200
    assert "password_changed" in _audit_actions(pg_dsn)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/admin/users"),
        ("post", "/admin/users/dr.someone/disable"),
        ("post", "/admin/users/dr.someone/revoke-tokens"),
    ],
)
def test_user_management_requires_admin(pg_dsn: str, method: str, path: str) -> None:
    with _make_client(pg_dsn) as client:
        doctor = _auth_headers(client, "dr.notadmin")
        response = getattr(client, method)(path, headers=doctor)
    assert response.status_code == 403


def test_unknown_user_is_404(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        response = client.post("/admin/users/dr.ghost/disable", headers=_admin_headers(client))
    assert response.status_code == 404
