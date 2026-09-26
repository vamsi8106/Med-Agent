import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from medagent.app import create_app
from medagent.core.config import Settings
from medagent.core.exceptions import MCPError, PatientNotFoundError
from medagent.core.models import AgentResult, PatientContext
from medagent.core.types import AgentRole
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore

_ADMIN_USERNAME = "admin"
_ADMIN_PASSWORD = "admin-pass!"


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


def _settings(pg_dsn: str) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="mock",
        postgres_dsn=pg_dsn,
        admin_bootstrap_username=_ADMIN_USERNAME,
        admin_bootstrap_password=_ADMIN_PASSWORD,
    )


@contextmanager
def _make_client(pg_dsn: str) -> Iterator[TestClient]:
    # AppState (and therefore VectorStore/EmbeddingModel construction) only
    # happens inside FastAPI's lifespan, which TestClient triggers on
    # __enter__ -- so these patches must still be active at that point, not
    # just while create_app() itself runs.
    with (
        patch("medagent.app.EmbeddingModel", _FakeEmbeddingModel),
        patch("medagent.app.VectorStore", _FakeVectorStore),
        patch("medagent.app.EvidenceAgent", _FakeEvidenceAgent),
        TestClient(create_app(_settings(pg_dsn))) as client,
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
        json={"username": username, "password": "s3cret!"},
        headers=_admin_headers(client),
    )
    token = _login(client, username, "s3cret!")
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
            json={"username": "dr.other", "password": "s3cret!"},
            headers=doctor_headers,
        )
    assert response.status_code == 403


def test_admin_endpoint_rejects_unauthenticated(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        response = client.post("/admin/users", json={"username": "dr.new", "password": "s3cret!"})
    assert response.status_code == 401


def test_admin_creates_user_and_they_can_log_in(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        create_resp = client.post(
            "/admin/users",
            json={"username": "dr.beta", "password": "s3cret!"},
            headers=_admin_headers(client),
        )
        assert create_resp.status_code == 200
        assert create_resp.json()["username"] == "dr.beta"
        assert create_resp.json()["role"] == "doctor"

        login_resp = client.post("/auth/token", data={"username": "dr.beta", "password": "s3cret!"})
    assert login_resp.status_code == 200


def test_login_with_wrong_password_returns_401(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        client.post(
            "/admin/users",
            json={"username": "dr.gamma", "password": "s3cret!"},
            headers=_admin_headers(client),
        )
        response = client.post("/auth/token", data={"username": "dr.gamma", "password": "wrong"})
    assert response.status_code == 401


def test_duplicate_username_returns_409(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        headers = _admin_headers(client)
        client.post(
            "/admin/users", json={"username": "dr.delta", "password": "s3cret!"}, headers=headers
        )
        response = client.post(
            "/admin/users", json={"username": "dr.delta", "password": "other"}, headers=headers
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
        with patch("medagent.app.run_patient_assessment", AsyncMock(return_value="# Report")):
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
        with patch("medagent.app.run_patient_assessment", AsyncMock(return_value="# Report")):
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
        with patch("medagent.app.run_drug_check", AsyncMock(return_value="No major concerns.")):
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
        with patch("medagent.app.run_drug_check", AsyncMock(side_effect=MCPError("upstream down"))):
            response = client.post(
                "/drug-check",
                json={"patient_id": "P-TEST-803", "new_drug": "Glimepiride"},
                headers=headers,
            )

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream down"}


def test_patient_not_found_error_maps_to_404(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        headers = _auth_headers(client)
        with patch(
            "medagent.app.run_drug_check",
            AsyncMock(side_effect=PatientNotFoundError("no such patient")),
        ):
            response = client.post(
                "/drug-check",
                json={"patient_id": "P-TEST-804", "new_drug": "Glimepiride"},
                headers=headers,
            )

    assert response.status_code == 404
    assert response.json() == {"detail": "no such patient"}


def test_websocket_without_token_is_rejected(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        try:
            with client.websocket_connect("/ws/P-TEST-999"):
                pass
        except Exception:
            pass
        else:
            raise AssertionError("expected the connection to be rejected")


def _create_patient(client: TestClient, token: str, patient_id: str, name: str) -> None:
    response = client.post(
        "/patients",
        json={"id": patient_id, "name": name, "age": 70, "sex": "M"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


def _ws_url(patient_id: str, token: str, resume: bool = False) -> str:
    return f"/ws/{patient_id}?token={token}" + ("&resume=1" if resume else "")


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
        with client.websocket_connect(_ws_url("P-TEST-803", token)) as websocket:
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
        with client.websocket_connect(_ws_url("P-TEST-804", token)) as websocket:
            _draft(websocket, "any updates?")
            websocket.send_text("reject")
            assert websocket.receive_json() == {"type": "rejected"}

    assert _visits(pg_dsn, "P-TEST-804") == []


def test_websocket_chat_edited_text_is_saved_instead(pg_dsn: str) -> None:
    with _make_client(pg_dsn) as client:
        token = _admin_token(client)
        _create_patient(client, token, "P-TEST-805", "Patient Epsilon")
        with client.websocket_connect(_ws_url("P-TEST-805", token)) as websocket:
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
        with client.websocket_connect(_ws_url("P-TEST-806", token)) as websocket:
            _, pending = _draft(websocket, "how is she doing?")
        # connection dropped with the draft still awaiting a decision
        assert _visits(pg_dsn, "P-TEST-806") == []

        with client.websocket_connect(_ws_url("P-TEST-806", token, resume=True)) as websocket:
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
        with client.websocket_connect(_ws_url("P-TEST-807", token)) as websocket:
            _draft(websocket, "first question")

        with client.websocket_connect(_ws_url("P-TEST-807", token)) as websocket:
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
        with client.websocket_connect(_ws_url("P-TEST-808", admin_token)) as websocket:
            _draft(websocket, "any updates?")

        headers = _auth_headers(client, "dr.intruder")
        intruder_token = headers["Authorization"].removeprefix("Bearer ")
        with client.websocket_connect(
            _ws_url("P-TEST-808", intruder_token, resume=True)
        ) as websocket:
            websocket.send_text("any updates?")
            frame = websocket.receive_json()
            # no draft handed over, and the patient isn't theirs to draft for
            assert frame["type"] == "error"
            assert "P-TEST-808" in frame["detail"]

    assert _visits(pg_dsn, "P-TEST-808") == []
