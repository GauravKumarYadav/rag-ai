"""
Live Integration Tests for Chat Application API Endpoints.

These tests run against the actual running containers to verify:
1. All services are up and accessible
2. All API endpoints return expected responses
3. Authentication flow works correctly
4. Document upload/search works
5. Conversation management works
6. Client management works

Run with: pytest tests/test_live_endpoints.py -v
Or run specific test: pytest tests/test_live_endpoints.py::TestHealthAndStatus -v
"""

import os
import time
import pytest
import requests
from typing import Optional

# Configuration
BASE_URL = os.getenv("API_BASE_URL", "http://localhost")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHROMADB_URL = os.getenv("CHROMADB_URL", "http://localhost:8020")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Test credentials
TEST_USERNAME = os.getenv("TEST_USERNAME", "admin")
TEST_PASSWORD = os.getenv("TEST_PASSWORD", "admin123")

# Request timeout
TIMEOUT = 30


class TestContainerHealth:
    """Test that all required containers are running and accessible."""

    def test_nginx_gateway_accessible(self):
        """Test that nginx gateway is accessible on port 80."""
        try:
            response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
            assert response.status_code == 200, f"Nginx returned {response.status_code}"
        except requests.ConnectionError as e:
            pytest.fail(f"Nginx gateway not accessible at {BASE_URL}: {e}")

    def test_chat_api_health(self):
        """Test that chat-api backend is healthy via /health endpoint."""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            assert response.status_code == 200, f"Health check returned {response.status_code}"
            data = response.json()
            assert "status" in data or "healthy" in str(data).lower(), f"Unexpected health response: {data}"
        except requests.ConnectionError as e:
            pytest.fail(f"Chat API health endpoint not accessible: {e}")

    def test_ollama_accessible(self):
        """Test that Ollama LLM service is accessible."""
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=TIMEOUT)
            assert response.status_code == 200, f"Ollama returned {response.status_code}"
            data = response.json()
            assert "models" in data, f"Unexpected Ollama response: {data}"
        except requests.ConnectionError as e:
            pytest.fail(f"Ollama not accessible at {OLLAMA_URL}: {e}")

    def test_chromadb_accessible(self):
        """Test that ChromaDB vector database is accessible."""
        try:
            response = requests.get(f"{CHROMADB_URL}/api/v1/heartbeat", timeout=TIMEOUT)
            assert response.status_code == 200, f"ChromaDB returned {response.status_code}"
        except requests.ConnectionError as e:
            pytest.fail(f"ChromaDB not accessible at {CHROMADB_URL}: {e}")

    def test_redis_accessible(self):
        """Test that Redis is accessible."""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((REDIS_HOST, REDIS_PORT))
            sock.close()
            assert result == 0, f"Redis not accessible on {REDIS_HOST}:{REDIS_PORT}"
        except Exception as e:
            pytest.fail(f"Redis connection test failed: {e}")


class TestAuthentication:
    """Test authentication endpoints."""

    def test_login_with_valid_credentials(self):
        """Test login with valid credentials returns access token."""
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Login failed: {response.text}"
        data = response.json()
        assert "access_token" in data, f"No access_token in response: {data}"
        assert "token_type" in data, f"No token_type in response: {data}"
        assert data["token_type"] == "bearer"

    def test_login_with_invalid_credentials(self):
        """Test login with invalid credentials returns 401."""
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": "invalid_user", "password": "wrong_password"},
            timeout=TIMEOUT,
        )
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"

    def test_auth_me_without_token(self):
        """Test /auth/me without token returns 401 or 403."""
        response = requests.get(f"{BASE_URL}/auth/me", timeout=TIMEOUT)
        assert response.status_code in [401, 403], f"Expected 401/403, got {response.status_code}"

    def test_auth_me_with_valid_token(self):
        """Test /auth/me with valid token returns user info."""
        # First login
        login_response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
            timeout=TIMEOUT,
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Then check /auth/me
        response = requests.get(
            f"{BASE_URL}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Auth me failed: {response.text}"
        data = response.json()
        assert "username" in data, f"No username in response: {data}"
        assert "user_id" in data, f"No user_id in response: {data}"


@pytest.fixture
def auth_token():
    """Get authentication token for protected endpoints."""
    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        pytest.skip(f"Could not get auth token: {response.text}")
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(auth_token):
    """Get authentication headers for protected endpoints."""
    return {"Authorization": f"Bearer {auth_token}"}


class TestHealthAndStatus:
    """Test health and status endpoints."""

    def test_status_endpoint(self, auth_headers):
        """Test /status returns system status."""
        response = requests.get(f"{BASE_URL}/status", timeout=TIMEOUT)
        assert response.status_code == 200, f"Status failed: {response.text}"
        data = response.json()
        assert "status" in data, f"No status in response: {data}"
        assert data["status"] == "ok", f"Status not ok: {data}"
        assert "model" in data, f"No model in response: {data}"
        assert "provider" in data, f"No provider in response: {data}"

    def test_ingest_status(self, auth_headers):
        """Test /ingest/status returns document count."""
        response = requests.get(f"{BASE_URL}/ingest/status", timeout=TIMEOUT)
        assert response.status_code == 200, f"Ingest status failed: {response.text}"
        data = response.json()
        assert "documents_indexed" in data, f"No documents_indexed: {data}"

    def test_memory_status(self, auth_headers):
        """Test /memory/status returns memory count."""
        response = requests.get(f"{BASE_URL}/memory/status", timeout=TIMEOUT)
        assert response.status_code == 200, f"Memory status failed: {response.text}"
        data = response.json()
        assert "memories_indexed" in data, f"No memories_indexed: {data}"

    def test_health_endpoint(self):
        """Test /health returns health status."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200, f"Health failed: {response.text}"


class TestConversations:
    """Test conversation management endpoints."""

    def test_list_conversations(self, auth_headers):
        """Test GET /conversations returns conversation list."""
        response = requests.get(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"List conversations failed: {response.text}"
        data = response.json()
        assert "conversations" in data, f"No conversations key: {data}"
        assert isinstance(data["conversations"], list)

    def test_create_conversation(self, auth_headers):
        """Test POST /conversations creates a new conversation."""
        response = requests.post(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            json={"title": "Test Conversation"},
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Create conversation failed: {response.text}"
        data = response.json()
        assert "id" in data, f"No id in response: {data}"
        assert "title" in data, f"No title in response: {data}"
        return data["id"]

    def test_get_conversation(self, auth_headers):
        """Test GET /conversations/{id} returns conversation details."""
        # First create a conversation
        create_response = requests.post(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            json={"title": "Test Get Conversation"},
            timeout=TIMEOUT,
        )
        assert create_response.status_code == 200
        conv_id = create_response.json()["id"]

        # Then get it
        response = requests.get(
            f"{BASE_URL}/conversations/{conv_id}",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Get conversation failed: {response.text}"
        data = response.json()
        assert "conversation_id" in data, f"No conversation_id: {data}"
        assert "messages" in data, f"No messages: {data}"

    def test_delete_conversation(self, auth_headers):
        """Test DELETE /conversations/{id} deletes a conversation."""
        # First create a conversation
        create_response = requests.post(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            json={"title": "Test Delete Conversation"},
            timeout=TIMEOUT,
        )
        assert create_response.status_code == 200
        conv_id = create_response.json()["id"]

        # Then delete it
        response = requests.delete(
            f"{BASE_URL}/conversations/{conv_id}",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Delete conversation failed: {response.text}"


class TestClients:
    """Test client management endpoints."""

    def test_list_clients(self, auth_headers):
        """Test GET /clients returns client list."""
        response = requests.get(
            f"{BASE_URL}/clients",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"List clients failed: {response.text}"
        data = response.json()
        assert "clients" in data, f"No clients key: {data}"
        assert isinstance(data["clients"], list)

    def test_create_client(self, auth_headers):
        """Test POST /clients creates a new client."""
        unique_name = f"Test Client {int(time.time())}"
        response = requests.post(
            f"{BASE_URL}/clients",
            headers=auth_headers,
            json={"name": unique_name},
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Create client failed: {response.text}"
        data = response.json()
        assert "id" in data, f"No id in response: {data}"
        assert "name" in data, f"No name in response: {data}"
        assert data["name"] == unique_name
        return data["id"]

    def test_get_client(self, auth_headers):
        """Test GET /clients/{id} returns client details."""
        # First create a client
        unique_name = f"Test Get Client {int(time.time())}"
        create_response = requests.post(
            f"{BASE_URL}/clients",
            headers=auth_headers,
            json={"name": unique_name},
            timeout=TIMEOUT,
        )
        assert create_response.status_code == 200
        client_id = create_response.json()["id"]

        # Then get it
        response = requests.get(
            f"{BASE_URL}/clients/{client_id}",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Get client failed: {response.text}"
        data = response.json()
        assert "id" in data, f"No id: {data}"
        assert data["id"] == client_id

    def test_get_client_stats(self, auth_headers):
        """Test GET /clients/{id}/stats returns client stats."""
        # First create a client
        unique_name = f"Test Stats Client {int(time.time())}"
        create_response = requests.post(
            f"{BASE_URL}/clients",
            headers=auth_headers,
            json={"name": unique_name},
            timeout=TIMEOUT,
        )
        assert create_response.status_code == 200
        client_id = create_response.json()["id"]

        # Then get stats
        response = requests.get(
            f"{BASE_URL}/clients/{client_id}/stats",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Get client stats failed: {response.text}"
        data = response.json()
        assert "client_id" in data, f"No client_id: {data}"
        assert "document_count" in data, f"No document_count: {data}"


class TestDocuments:
    """Test document management endpoints."""

    def test_list_documents(self, auth_headers):
        """Test GET /documents returns document list."""
        response = requests.get(
            f"{BASE_URL}/documents",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"List documents failed: {response.text}"
        data = response.json()
        assert isinstance(data, list), f"Expected list, got: {type(data)}"

    def test_documents_stats(self, auth_headers):
        """Test GET /documents/stats returns document statistics."""
        response = requests.get(
            f"{BASE_URL}/documents/stats",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Documents stats failed: {response.text}"
        data = response.json()
        assert "documents_indexed" in data, f"No documents_indexed: {data}"

    def test_documents_formats(self, auth_headers):
        """Test GET /documents/formats returns supported formats."""
        response = requests.get(
            f"{BASE_URL}/documents/formats",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Documents formats failed: {response.text}"
        data = response.json()
        assert isinstance(data, (list, dict)), f"Expected list or dict: {data}"

    def test_documents_search(self, auth_headers):
        """Test POST /documents/search searches documents."""
        response = requests.post(
            f"{BASE_URL}/documents/search",
            headers=auth_headers,
            json={"query": "test search query", "top_k": 5},
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Documents search failed: {response.text}"
        data = response.json()
        assert "results" in data, f"No results: {data}"

    def test_upload_text_document(self, auth_headers):
        """Test POST /documents/upload uploads a text file."""
        # Create a simple text file content
        file_content = b"This is a test document for the RAG chatbot system."
        files = {"files": ("test_document.txt", file_content, "text/plain")}
        
        response = requests.post(
            f"{BASE_URL}/documents/upload",
            headers={"Authorization": auth_headers["Authorization"]},
            files=files,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Upload document failed: {response.text}"
        data = response.json()
        assert "message" in data, f"No message in response: {data}"


class TestChat:
    """Test chat endpoints."""

    def test_chat_endpoint(self, auth_headers):
        """Test POST /chat sends a message and gets a response."""
        # First create a conversation
        create_response = requests.post(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            json={"title": "Test Chat Conversation"},
            timeout=TIMEOUT,
        )
        assert create_response.status_code == 200
        conv_id = create_response.json()["id"]

        # Then send a chat message
        response = requests.post(
            f"{BASE_URL}/chat",
            headers=auth_headers,
            json={
                "conversation_id": conv_id,
                "message": "Hello, this is a test message.",
                "stream": False,
            },
            timeout=60,  # Longer timeout for LLM response
        )
        assert response.status_code == 200, f"Chat failed: {response.text}"
        data = response.json()
        assert "response" in data, f"No response in chat response: {data}"


class TestModels:
    """Test model management endpoints."""

    def test_list_models(self, auth_headers):
        """Test GET /models returns available models."""
        response = requests.get(
            f"{BASE_URL}/models",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"List models failed: {response.text}"
        data = response.json()
        assert "models" in data or isinstance(data, list), f"Unexpected response: {data}"

    def test_current_model(self, auth_headers):
        """Test GET /models/current returns current model config."""
        response = requests.get(
            f"{BASE_URL}/models/current",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Current model failed: {response.text}"
        data = response.json()
        assert "provider" in data or "model" in data, f"No provider/model info: {data}"

    def test_list_providers(self, auth_headers):
        """Test GET /models/providers returns supported providers."""
        response = requests.get(
            f"{BASE_URL}/models/providers",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"List providers failed: {response.text}"


class TestAdmin:
    """Test admin endpoints (requires superuser)."""

    def test_admin_stats(self, auth_headers):
        """Test GET /admin/stats returns system statistics."""
        response = requests.get(
            f"{BASE_URL}/admin/stats",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        # Admin endpoints may require superuser, accept 200 or 403
        assert response.status_code in [200, 403], f"Admin stats failed: {response.text}"

    def test_admin_config(self, auth_headers):
        """Test GET /admin/config returns system configuration."""
        response = requests.get(
            f"{BASE_URL}/admin/config",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code in [200, 403], f"Admin config failed: {response.text}"

    def test_admin_users(self, auth_headers):
        """Test GET /admin/users returns user list."""
        response = requests.get(
            f"{BASE_URL}/admin/users",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code in [200, 403], f"Admin users failed: {response.text}"

    def test_admin_audit_logs(self, auth_headers):
        """Test GET /admin/audit-logs returns audit logs."""
        response = requests.get(
            f"{BASE_URL}/admin/audit-logs",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code in [200, 403], f"Audit logs failed: {response.text}"


class TestEvaluation:
    """Test evaluation endpoints."""

    def test_list_datasets(self, auth_headers):
        """Test GET /evaluation/datasets returns dataset list."""
        response = requests.get(
            f"{BASE_URL}/evaluation/datasets",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"List datasets failed: {response.text}"

    def test_list_evaluation_runs(self, auth_headers):
        """Test GET /evaluation/runs returns evaluation runs."""
        response = requests.get(
            f"{BASE_URL}/evaluation/runs",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"List runs failed: {response.text}"


class TestEndpointSummary:
    """Summary test to verify all critical endpoints at once."""

    def test_all_critical_endpoints(self, auth_headers):
        """Quick smoke test of all critical endpoints."""
        endpoints = [
            ("GET", "/status", None),
            ("GET", "/health", None),
            ("GET", "/ingest/status", None),
            ("GET", "/memory/status", None),
            ("GET", "/conversations", auth_headers),
            ("GET", "/clients", auth_headers),
            ("GET", "/documents", auth_headers),
            ("GET", "/documents/stats", auth_headers),
            ("GET", "/models", auth_headers),
            ("GET", "/models/current", auth_headers),
            ("GET", "/evaluation/datasets", auth_headers),
            ("GET", "/evaluation/runs", auth_headers),
        ]

        failed = []
        for method, endpoint, headers in endpoints:
            try:
                if method == "GET":
                    response = requests.get(
                        f"{BASE_URL}{endpoint}",
                        headers=headers,
                        timeout=TIMEOUT,
                    )
                else:
                    response = requests.post(
                        f"{BASE_URL}{endpoint}",
                        headers=headers,
                        timeout=TIMEOUT,
                    )

                if response.status_code >= 400:
                    failed.append(f"{method} {endpoint}: {response.status_code}")
            except Exception as e:
                failed.append(f"{method} {endpoint}: {str(e)}")

        if failed:
            pytest.fail(f"Failed endpoints:\n" + "\n".join(failed))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
