import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from medagent.app import create_app
from medagent.core.config import Settings
from medagent.core.exceptions import MCPError, PatientNotFoundError
from medagent.core.models import PatientContext
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore

_ADMIN_USERNAME = "admin"
_ADMIN_PASSWORD = "admin-pass!"


class _FakeEmbeddingModel:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]


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


def test_websocket_chat_requires_approval_before_saving(pg_dsn: str) -> None:
    patient = PatientContext(
        id="P-TEST-803", name="Patient Gamma", age=70, sex="M", doctor_id="DR-TEST-001"
    )
    with _make_client(pg_dsn) as client:
        token = _admin_token(client)
        fake_followup = AsyncMock(return_value=("Follow-up report", patient))
        with patch("medagent.app.run_followup", fake_followup):
            with client.websocket_connect(f"/ws/P-TEST-803?token={token}") as websocket:
                websocket.send_text("any updates?")
                pending = websocket.receive_json()
                assert pending == {"type": "pending_approval", "report": "Follow-up report"}

                websocket.send_text("approve")
                saved = websocket.receive_json()
                assert saved == {"type": "saved", "report": "Follow-up report"}

        headers = {"Authorization": f"Bearer {token}"}
        get_resp = client.get("/patients/P-TEST-803", headers=headers)
    assert get_resp.status_code == 200

    store = PatientStore(PersistentStore(pg_dsn))
    loaded = asyncio.run(store.get_patient("P-TEST-803"))
    assert loaded is not None
    assert len(loaded.visits) == 1
    assert loaded.visits[0].chief_complaint == "any updates?"


def test_websocket_chat_reject_does_not_save(pg_dsn: str) -> None:
    patient = PatientContext(
        id="P-TEST-804", name="Patient Delta", age=65, sex="F", doctor_id="DR-TEST-001"
    )
    with _make_client(pg_dsn) as client:
        token = _admin_token(client)
        fake_followup = AsyncMock(return_value=("Draft report", patient))
        with patch("medagent.app.run_followup", fake_followup):
            with client.websocket_connect(f"/ws/P-TEST-804?token={token}") as websocket:
                websocket.send_text("any updates?")
                websocket.receive_json()

                websocket.send_text("reject")
                decision = websocket.receive_json()
                assert decision == {"type": "rejected"}

        headers = {"Authorization": f"Bearer {token}"}
        get_resp = client.get("/patients/P-TEST-804", headers=headers)
    assert get_resp.status_code == 404


def test_websocket_chat_edited_text_is_saved_instead(pg_dsn: str) -> None:
    patient = PatientContext(
        id="P-TEST-805", name="Patient Epsilon", age=45, sex="F", doctor_id="DR-TEST-001"
    )
    with _make_client(pg_dsn) as client:
        token = _admin_token(client)
        fake_followup = AsyncMock(return_value=("Draft report", patient))
        with patch("medagent.app.run_followup", fake_followup):
            with client.websocket_connect(f"/ws/P-TEST-805?token={token}") as websocket:
                websocket.send_text("any updates?")
                websocket.receive_json()

                websocket.send_text("Edited: monitor renal function closely.")
                saved = websocket.receive_json()
                assert saved == {
                    "type": "saved",
                    "report": "Edited: monitor renal function closely.",
                }
