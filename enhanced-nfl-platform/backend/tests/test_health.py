import os

from fastapi.testclient import TestClient

os.environ.setdefault("TEST_MODE", "true")

from main import app  # noqa: E402


def create_client():
    return TestClient(app)


def test_root_endpoint():
    with create_client() as client:
        response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "NFL AI/ML Platform API"


def test_health_endpoint():
    with create_client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["ml_pipeline"] in {"ready", "loading"}
    assert body["rag_system"] in {"ready", "loading"}
