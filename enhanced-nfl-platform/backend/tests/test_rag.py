from contextlib import contextmanager

from fastapi import status

from main import app
from app.core.rag_system import get_rag_system


class StubRAG:
    def __init__(self, *, should_fail: bool = False):
        self.should_fail = should_fail

    async def query(self, question: str, top_k: int = 5):
        if self.should_fail:
            raise RuntimeError("rag failure")
        return {
            "question": question,
            "answer": "Sample answer",
            "relevant_docs": [{"id": 1, "score": 0.9}],
            "confidence": 0.82,
        }

    async def get_knowledge_stats(self):
        if self.should_fail:
            raise RuntimeError("stats failure")
        return {
            "total_documents": 10,
            "vector_db_status": "ready",
            "embedding_model": "stub",
            "index_name": "stub-index",
        }

    async def add_document(self, content: str, metadata: dict):
        if self.should_fail:
            raise RuntimeError("add failure")
        return None


@contextmanager
def override_rag(system):
    app.dependency_overrides[get_rag_system] = lambda: system
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_rag_system, None)


def test_rag_query_success(client):
    with override_rag(StubRAG()):
        response = client.post(
            "/api/v1/rag/query",
            json={"question": "Who are the top quarterbacks?", "top_k": 3},
        )
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["answer"] == "Sample answer"
    assert body["confidence"] == 0.82


def test_rag_query_failure(client):
    with override_rag(StubRAG(should_fail=True)):
        response = client.post("/api/v1/rag/query", json={"question": "Fail", "top_k": 1})
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_rag_stats_success(client):
    with override_rag(StubRAG()):
        response = client.get("/api/v1/rag/stats")
    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload["total_documents"] == 10
    assert payload["vector_db_status"] == "ready"
