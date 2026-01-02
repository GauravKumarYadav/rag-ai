import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_documents_stats(client):
    response = client.get("/documents/stats")
    assert response.status_code == 200
    data = response.json()
    assert "documents_indexed" in data
    assert "memories_indexed" in data


def test_documents_search(client):
    response = client.post(
        "/documents/search",
        json={"query": "test query", "top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_conversations_list(client):
    response = client.get("/conversations")
    assert response.status_code == 200
    data = response.json()
    assert "conversations" in data


def test_conversation_get_nonexistent(client):
    response = client.get("/conversations/nonexistent-id")
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "nonexistent-id"
    assert data["messages"] == []
    assert data["message_count"] == 0


def test_ingest_status(client):
    response = client.get("/ingest/status")
    assert response.status_code == 200
    data = response.json()
    assert "documents_indexed" in data


def test_memory_status(client):
    response = client.get("/memory/status")
    assert response.status_code == 200
    data = response.json()
    assert "memories_indexed" in data
