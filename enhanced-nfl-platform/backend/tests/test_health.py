def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "NFL AI/ML Platform API"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["ml_pipeline"] in {"ready", "loading"}
    assert body["rag_system"] in {"ready", "loading"}
