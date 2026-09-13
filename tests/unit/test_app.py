from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from medagent.app import create_app
from medagent.core.config import Settings


class _FakeEmbeddingModel:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="mock",
        database_path=str(tmp_path / "test.db"),
        chroma_persist_dir=str(tmp_path / "chroma"),
    )


def _make_client(tmp_path: Path) -> TestClient:
    with patch("medagent.app.EmbeddingModel", _FakeEmbeddingModel):
        app = create_app(_settings(tmp_path))
    return TestClient(app)


def test_health_endpoint(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert b"medagent_http_requests_total" in response.content


def test_create_and_get_patient(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        payload = {"id": "P-TEST-800", "name": "Patient Alpha", "age": 55, "sex": "F"}
        create_resp = client.post("/patients", json=payload)
        assert create_resp.status_code == 200

        get_resp = client.get("/patients/P-TEST-800")

    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Patient Alpha"


def test_get_missing_patient_returns_404(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        response = client.get("/patients/P-MISSING")
    assert response.status_code == 404


def test_assess_endpoint_returns_report(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        client.post(
            "/patients", json={"id": "P-TEST-801", "name": "Patient Beta", "age": 60, "sex": "M"}
        )
        with patch("medagent.app.run_patient_assessment", AsyncMock(return_value="# Report")):
            response = client.post("/patients/P-TEST-801/assess", json={"message": "hi"})

    assert response.status_code == 200
    assert response.json() == {"report": "# Report"}


def test_drug_check_endpoint(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        with patch("medagent.app.run_drug_check", AsyncMock(return_value="No major concerns.")):
            response = client.post(
                "/drug-check", json={"patient_id": "P-TEST-802", "new_drug": "Glimepiride"}
            )

    assert response.status_code == 200
    assert response.json() == {"answer": "No major concerns."}


def test_websocket_chat_returns_report(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        with patch("medagent.app.run_followup", AsyncMock(return_value="Follow-up report")):
            with client.websocket_connect("/ws/P-TEST-803") as websocket:
                websocket.send_text("any updates?")
                message = websocket.receive_text()

    assert message == "Follow-up report"
