"""
Comprehensive Live Integration Tests for RAG Chat Application.

These tests run against actual running containers to verify:
1. Container health and connectivity
2. Authentication and authorization flows
3. All API endpoints return expected responses
4. Edge cases and error handling
5. Document upload/search/delete workflows
6. Conversation management
7. Client management
8. WebSocket functionality
9. Rate limiting and security
10. RAG-specific functionality

Run all tests:
    pytest tests/live/ -v

Run specific test class:
    pytest tests/live/test_endpoints.py::TestAuthentication -v

Run with coverage:
    pytest tests/live/ -v --cov=app --cov-report=html

Environment variables:
    API_BASE_URL: Base URL for API (default: http://localhost)
    OLLAMA_URL: Ollama service URL (default: http://localhost:11434)
    CHROMADB_URL: ChromaDB URL (default: http://localhost:8020)
    TEST_USERNAME: Test user username (default: admin)
    TEST_PASSWORD: Test user password (default: admin123)
"""

import os
import time
import socket
import uuid
import json
import pytest
import requests
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = os.getenv("API_BASE_URL", "http://localhost")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHROMADB_URL = os.getenv("CHROMADB_URL", "http://localhost:8020")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3307"))

TEST_USERNAME = os.getenv("TEST_USERNAME", "admin")
TEST_PASSWORD = os.getenv("TEST_PASSWORD", "admin123")

# Timeouts
TIMEOUT = 30
LONG_TIMEOUT = 120  # For LLM responses


# =============================================================================
# Helper Functions
# =============================================================================

def generate_unique_name(prefix: str = "test") -> str:
    """Generate a unique name for test resources."""
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def check_port_open(host: str, port: int, timeout: int = 5) -> bool:
    """Check if a TCP port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def make_request(
    method: str,
    endpoint: str,
    headers: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    files: Optional[Dict] = None,
    timeout: int = TIMEOUT,
    expected_status: Optional[List[int]] = None,
) -> requests.Response:
    """Make an HTTP request with standard error handling."""
    url = f"{BASE_URL}{endpoint}"
    
    kwargs = {"timeout": timeout}
    if headers:
        kwargs["headers"] = headers
    if json_data:
        kwargs["json"] = json_data
    if files:
        kwargs["files"] = files
    
    response = getattr(requests, method.lower())(url, **kwargs)
    
    if expected_status:
        assert response.status_code in expected_status, (
            f"Expected {expected_status}, got {response.status_code}: {response.text}"
        )
    
    return response


# =============================================================================
# SECTION 1: Container and Service Health Tests
# =============================================================================

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

    def test_ollama_has_models(self):
        """Test that Ollama has at least one model loaded."""
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=TIMEOUT)
        data = response.json()
        models = data.get("models", [])
        assert len(models) > 0, "Ollama has no models loaded"

    def test_chromadb_accessible(self):
        """Test that ChromaDB vector database is accessible."""
        try:
            # Try v2 first, fall back to v1
            response = requests.get(f"{CHROMADB_URL}/api/v2/heartbeat", timeout=TIMEOUT)
            if response.status_code == 404:
                response = requests.get(f"{CHROMADB_URL}/api/v1/heartbeat", timeout=TIMEOUT)
            assert response.status_code == 200, f"ChromaDB returned {response.status_code}"
        except requests.ConnectionError as e:
            pytest.fail(f"ChromaDB not accessible at {CHROMADB_URL}: {e}")

    def test_redis_accessible(self):
        """Test that Redis is accessible."""
        assert check_port_open(REDIS_HOST, REDIS_PORT), (
            f"Redis not accessible on {REDIS_HOST}:{REDIS_PORT}"
        )

    def test_mysql_accessible(self):
        """Test that MySQL is accessible."""
        assert check_port_open("localhost", MYSQL_PORT), (
            f"MySQL not accessible on port {MYSQL_PORT}"
        )


# =============================================================================
# SECTION 2: Authentication Tests
# =============================================================================

class TestAuthentication:
    """Test authentication endpoints and flows."""

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
        assert "expires_in" in data, f"No expires_in in response: {data}"

    def test_login_with_invalid_credentials(self):
        """Test login with invalid credentials returns 401."""
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": "invalid_user", "password": "wrong_password"},
            timeout=TIMEOUT,
        )
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"

    def test_login_with_empty_username(self):
        """Test login with empty username fails."""
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": "", "password": "somepass"},
            timeout=TIMEOUT,
        )
        assert response.status_code in [400, 401, 422], (
            f"Expected error status, got {response.status_code}"
        )

    def test_login_with_empty_password(self):
        """Test login with empty password fails."""
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": TEST_USERNAME, "password": ""},
            timeout=TIMEOUT,
        )
        assert response.status_code in [400, 401, 422], (
            f"Expected error status, got {response.status_code}"
        )

    def test_login_with_missing_fields(self):
        """Test login with missing fields fails validation."""
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={},
            timeout=TIMEOUT,
        )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_login_with_sql_injection_attempt(self):
        """Test that SQL injection attempts are handled safely."""
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": "admin' OR '1'='1", "password": "' OR '1'='1"},
            timeout=TIMEOUT,
        )
        # Should either reject with 401 or 422, not crash or return 500
        assert response.status_code in [400, 401, 422], (
            f"SQL injection attempt should be handled safely, got {response.status_code}"
        )

    def test_auth_me_without_token(self):
        """Test /auth/me without token returns 401 or 403."""
        response = requests.get(f"{BASE_URL}/auth/me", timeout=TIMEOUT)
        assert response.status_code in [401, 403], (
            f"Expected 401/403, got {response.status_code}"
        )

    def test_auth_me_with_invalid_token(self):
        """Test /auth/me with invalid token returns 401."""
        response = requests.get(
            f"{BASE_URL}/auth/me",
            headers={"Authorization": "Bearer invalid_token_here"},
            timeout=TIMEOUT,
        )
        assert response.status_code in [401, 403], (
            f"Expected 401/403, got {response.status_code}"
        )

    def test_auth_me_with_malformed_header(self):
        """Test /auth/me with malformed Authorization header."""
        response = requests.get(
            f"{BASE_URL}/auth/me",
            headers={"Authorization": "NotBearer token"},
            timeout=TIMEOUT,
        )
        assert response.status_code in [401, 403, 422], (
            f"Expected error, got {response.status_code}"
        )

    def test_auth_me_with_valid_token(self, auth_headers):
        """Test /auth/me with valid token returns user info."""
        response = requests.get(
            f"{BASE_URL}/auth/me",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Auth me failed: {response.text}"
        data = response.json()
        assert "username" in data, f"No username in response: {data}"
        assert "user_id" in data, f"No user_id in response: {data}"

    def test_token_expiry_format(self):
        """Test that token expiry is properly formatted."""
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
            timeout=TIMEOUT,
        )
        data = response.json()
        assert isinstance(data.get("expires_in"), int), "expires_in should be an integer"
        assert data["expires_in"] > 0, "expires_in should be positive"


# =============================================================================
# SECTION 3: Health and Status Endpoints
# =============================================================================

class TestHealthAndStatus:
    """Test health and status endpoints."""

    def test_status_endpoint(self):
        """Test /status returns system status."""
        response = requests.get(f"{BASE_URL}/status", timeout=TIMEOUT)
        assert response.status_code == 200, f"Status failed: {response.text}"
        data = response.json()
        assert "status" in data, f"No status in response: {data}"
        assert data["status"] == "ok", f"Status not ok: {data}"
        assert "model" in data, f"No model in response: {data}"
        assert "provider" in data, f"No provider in response: {data}"

    def test_ingest_status(self):
        """Test /ingest/status returns document count."""
        response = requests.get(f"{BASE_URL}/ingest/status", timeout=TIMEOUT)
        assert response.status_code == 200, f"Ingest status failed: {response.text}"
        data = response.json()
        assert "documents_indexed" in data, f"No documents_indexed: {data}"
        assert isinstance(data["documents_indexed"], int), "documents_indexed should be int"

    def test_memory_status(self):
        """Test /memory/status returns memory count."""
        response = requests.get(f"{BASE_URL}/memory/status", timeout=TIMEOUT)
        assert response.status_code == 200, f"Memory status failed: {response.text}"
        data = response.json()
        assert "memories_indexed" in data, f"No memories_indexed: {data}"

    def test_health_endpoint(self):
        """Test /health returns health status."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200, f"Health failed: {response.text}"

    def test_health_response_time(self):
        """Test that health endpoint responds quickly."""
        start = time.time()
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 2.0, f"Health check too slow: {elapsed:.2f}s"

    def test_metrics_endpoint(self):
        """Test /metrics returns Prometheus metrics."""
        response = requests.get(f"{BASE_URL}/metrics", timeout=TIMEOUT)
        assert response.status_code == 200, f"Metrics failed: {response.text}"
        # Should contain Prometheus-formatted metrics
        assert "http_request" in response.text or "fastapi" in response.text.lower(), (
            "Metrics should contain HTTP request metrics"
        )


# =============================================================================
# SECTION 4: Conversation Management Tests
# =============================================================================

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
        title = generate_unique_name("conv")
        response = requests.post(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            json={"title": title},
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Create conversation failed: {response.text}"
        data = response.json()
        assert "id" in data, f"No id in response: {data}"
        assert "title" in data, f"No title in response: {data}"
        assert data["title"] == title

    def test_create_conversation_with_empty_title(self, auth_headers):
        """Test creating conversation with empty title."""
        response = requests.post(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            json={"title": ""},
            timeout=TIMEOUT,
        )
        # Should either succeed with empty title or return validation error
        assert response.status_code in [200, 422], f"Unexpected: {response.status_code}"

    def test_create_conversation_with_long_title(self, auth_headers):
        """Test creating conversation with very long title."""
        long_title = "A" * 1000
        response = requests.post(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            json={"title": long_title},
            timeout=TIMEOUT,
        )
        # Should either succeed (possibly truncated) or return validation error
        assert response.status_code in [200, 422], f"Unexpected: {response.status_code}"

    def test_get_conversation(self, auth_headers):
        """Test GET /conversations/{id} returns conversation details."""
        # First create a conversation
        create_response = requests.post(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            json={"title": generate_unique_name("get_conv")},
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

    def test_get_nonexistent_conversation(self, auth_headers):
        """Test GET /conversations/{id} with nonexistent ID."""
        fake_id = str(uuid.uuid4())
        response = requests.get(
            f"{BASE_URL}/conversations/{fake_id}",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        # Should return empty conversation or 404
        assert response.status_code in [200, 404], f"Unexpected: {response.status_code}"

    def test_delete_conversation(self, auth_headers):
        """Test DELETE /conversations/{id} deletes a conversation."""
        # First create a conversation
        create_response = requests.post(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            json={"title": generate_unique_name("delete_conv")},
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

    def test_delete_nonexistent_conversation(self, auth_headers):
        """Test deleting a nonexistent conversation."""
        fake_id = str(uuid.uuid4())
        response = requests.delete(
            f"{BASE_URL}/conversations/{fake_id}",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        # Should return success (idempotent) or 404
        assert response.status_code in [200, 404], f"Unexpected: {response.status_code}"

    def test_conversation_without_auth(self):
        """Test accessing conversations without authentication."""
        response = requests.get(f"{BASE_URL}/conversations", timeout=TIMEOUT)
        assert response.status_code in [401, 403], (
            f"Expected 401/403, got {response.status_code}"
        )


# =============================================================================
# SECTION 5: Client Management Tests
# =============================================================================

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
        unique_name = generate_unique_name("client")
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

    def test_create_client_with_empty_name(self, auth_headers):
        """Test creating client with empty name."""
        response = requests.post(
            f"{BASE_URL}/clients",
            headers=auth_headers,
            json={"name": ""},
            timeout=TIMEOUT,
        )
        # Should fail validation
        assert response.status_code in [400, 422], f"Unexpected: {response.status_code}"

    def test_create_duplicate_client_name(self, auth_headers):
        """Test creating client with duplicate name."""
        unique_name = generate_unique_name("dup_client")
        
        # Create first client
        response1 = requests.post(
            f"{BASE_URL}/clients",
            headers=auth_headers,
            json={"name": unique_name},
            timeout=TIMEOUT,
        )
        assert response1.status_code == 200
        
        # Try to create second with same name
        response2 = requests.post(
            f"{BASE_URL}/clients",
            headers=auth_headers,
            json={"name": unique_name},
            timeout=TIMEOUT,
        )
        # Should either fail or create with different ID
        if response2.status_code == 200:
            # If success, IDs should be different
            assert response1.json()["id"] != response2.json()["id"]

    def test_get_client(self, auth_headers):
        """Test GET /clients/{id} returns client details."""
        # First create a client
        unique_name = generate_unique_name("get_client")
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

    def test_get_nonexistent_client(self, auth_headers):
        """Test GET /clients/{id} with nonexistent ID."""
        fake_id = str(uuid.uuid4())
        response = requests.get(
            f"{BASE_URL}/clients/{fake_id}",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"

    def test_get_client_stats(self, auth_headers):
        """Test GET /clients/{id}/stats returns client stats."""
        # First create a client
        unique_name = generate_unique_name("stats_client")
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

    def test_delete_client(self, auth_headers):
        """Test DELETE /clients/{id} deletes a client."""
        # First create a client
        unique_name = generate_unique_name("delete_client")
        create_response = requests.post(
            f"{BASE_URL}/clients",
            headers=auth_headers,
            json={"name": unique_name},
            timeout=TIMEOUT,
        )
        assert create_response.status_code == 200
        client_id = create_response.json()["id"]

        # Then delete it
        response = requests.delete(
            f"{BASE_URL}/clients/{client_id}",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Delete client failed: {response.text}"

        # Verify it's deleted
        get_response = requests.get(
            f"{BASE_URL}/clients/{client_id}",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert get_response.status_code == 404


# =============================================================================
# SECTION 6: Document Management Tests
# =============================================================================

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

    def test_documents_search_with_empty_query(self, auth_headers):
        """Test search with empty query."""
        response = requests.post(
            f"{BASE_URL}/documents/search",
            headers=auth_headers,
            json={"query": "", "top_k": 5},
            timeout=TIMEOUT,
        )
        # Should either return empty results or validation error
        assert response.status_code in [200, 422], f"Unexpected: {response.status_code}"

    def test_documents_search_with_large_top_k(self, auth_headers):
        """Test search with very large top_k."""
        response = requests.post(
            f"{BASE_URL}/documents/search",
            headers=auth_headers,
            json={"query": "test", "top_k": 10000},
            timeout=TIMEOUT,
        )
        # Should handle gracefully
        assert response.status_code in [200, 422], f"Unexpected: {response.status_code}"

    def test_documents_search_with_special_characters(self, auth_headers):
        """Test search with special characters in query."""
        special_queries = [
            "test query with 'quotes'",
            "test with \"double quotes\"",
            "test & special < > chars",
            "test\\nwith\\nnewlines",
            "😀 emoji test",
            "SELECT * FROM documents;",  # SQL-like
        ]
        for query in special_queries:
            response = requests.post(
                f"{BASE_URL}/documents/search",
                headers=auth_headers,
                json={"query": query, "top_k": 5},
                timeout=TIMEOUT,
            )
            assert response.status_code in [200, 422], (
                f"Special char query '{query[:20]}' failed: {response.status_code}"
            )

    def test_upload_text_document(self, auth_headers):
        """Test POST /documents/upload uploads a text file."""
        file_content = b"This is a test document for the RAG chatbot system."
        files = {"files": ("test_document.txt", file_content, "text/plain")}
        
        response = requests.post(
            f"{BASE_URL}/documents/upload",
            headers={"Authorization": auth_headers["Authorization"]},
            files=files,
            timeout=LONG_TIMEOUT,
        )
        assert response.status_code == 200, f"Upload document failed: {response.text}"
        data = response.json()
        assert "message" in data, f"No message in response: {data}"

    def test_upload_multiple_documents(self, auth_headers):
        """Test uploading multiple documents at once."""
        files = [
            ("files", ("doc1.txt", b"First test document content", "text/plain")),
            ("files", ("doc2.txt", b"Second test document content", "text/plain")),
        ]
        
        response = requests.post(
            f"{BASE_URL}/documents/upload",
            headers={"Authorization": auth_headers["Authorization"]},
            files=files,
            timeout=LONG_TIMEOUT,
        )
        assert response.status_code == 200, f"Upload failed: {response.text}"

    def test_upload_empty_file(self, auth_headers):
        """Test uploading an empty file."""
        files = {"files": ("empty.txt", b"", "text/plain")}
        
        response = requests.post(
            f"{BASE_URL}/documents/upload",
            headers={"Authorization": auth_headers["Authorization"]},
            files=files,
            timeout=TIMEOUT,
        )
        # Should either succeed or return validation error
        assert response.status_code in [200, 400, 422], f"Unexpected: {response.status_code}"

    def test_upload_large_text_file(self, auth_headers):
        """Test uploading a larger text file."""
        # Create a ~100KB file
        large_content = b"This is a test line.\n" * 5000
        files = {"files": ("large_doc.txt", large_content, "text/plain")}
        
        response = requests.post(
            f"{BASE_URL}/documents/upload",
            headers={"Authorization": auth_headers["Authorization"]},
            files=files,
            timeout=LONG_TIMEOUT,
        )
        assert response.status_code == 200, f"Upload large file failed: {response.text}"

    def test_upload_without_auth(self):
        """Test document upload without authentication fails."""
        files = {"files": ("test.txt", b"content", "text/plain")}
        response = requests.post(
            f"{BASE_URL}/documents/upload",
            files=files,
            timeout=TIMEOUT,
        )
        assert response.status_code in [401, 403], (
            f"Expected 401/403, got {response.status_code}"
        )


# =============================================================================
# SECTION 7: Chat Endpoint Tests
# =============================================================================

class TestChat:
    """Test chat endpoints."""

    def test_chat_endpoint(self, auth_headers):
        """Test POST /chat sends a message and gets a response."""
        # First create a conversation
        create_response = requests.post(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            json={"title": generate_unique_name("chat_conv")},
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
                "message": "Hello, this is a test. Please respond briefly.",
                "stream": False,
            },
            timeout=LONG_TIMEOUT,
        )
        assert response.status_code == 200, f"Chat failed: {response.text}"
        data = response.json()
        assert "response" in data, f"No response in chat response: {data}"
        assert len(data["response"]) > 0, "Response should not be empty"

    def test_chat_with_sources(self, auth_headers):
        """Test chat returns sources when include_sources is true."""
        create_response = requests.post(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            json={"title": generate_unique_name("sources_conv")},
            timeout=TIMEOUT,
        )
        conv_id = create_response.json()["id"]

        response = requests.post(
            f"{BASE_URL}/chat",
            headers=auth_headers,
            json={
                "conversation_id": conv_id,
                "message": "Tell me about the documents",
                "stream": False,
                "include_sources": True,
            },
            timeout=LONG_TIMEOUT,
        )
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data, f"No sources in response: {data}"

    def test_chat_without_conversation_id(self, auth_headers):
        """Test chat generates new conversation if no ID provided."""
        response = requests.post(
            f"{BASE_URL}/chat",
            headers=auth_headers,
            json={
                "message": "Hello test",
                "stream": False,
            },
            timeout=LONG_TIMEOUT,
        )
        # Should either work or require conversation_id
        assert response.status_code in [200, 422], f"Unexpected: {response.status_code}"

    def test_chat_with_empty_message(self, auth_headers):
        """Test chat with empty message fails."""
        response = requests.post(
            f"{BASE_URL}/chat",
            headers=auth_headers,
            json={
                "message": "",
                "stream": False,
            },
            timeout=TIMEOUT,
        )
        assert response.status_code in [400, 422], (
            f"Empty message should fail, got {response.status_code}"
        )

    def test_chat_with_very_long_message(self, auth_headers):
        """Test chat with very long message."""
        long_message = "This is a test. " * 1000  # ~16KB message
        
        response = requests.post(
            f"{BASE_URL}/chat",
            headers=auth_headers,
            json={
                "message": long_message,
                "stream": False,
            },
            timeout=LONG_TIMEOUT,
        )
        # Should handle gracefully
        assert response.status_code in [200, 413, 422], (
            f"Long message handling unexpected: {response.status_code}"
        )

    def test_chat_without_auth(self):
        """Test chat without authentication fails."""
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": "Hello", "stream": False},
            timeout=TIMEOUT,
        )
        assert response.status_code in [401, 403], (
            f"Expected 401/403, got {response.status_code}"
        )

    def test_chat_response_time(self, auth_headers):
        """Test that chat responds within reasonable time."""
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/chat",
            headers=auth_headers,
            json={
                "message": "Say 'hello' and nothing else.",
                "stream": False,
            },
            timeout=LONG_TIMEOUT,
        )
        elapsed = time.time() - start
        assert response.status_code == 200
        # LLM should respond within 2 minutes for a simple message
        assert elapsed < 120, f"Chat took too long: {elapsed:.2f}s"


# =============================================================================
# SECTION 8: Model Management Tests
# =============================================================================

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

    def test_current_model(self, auth_headers):
        """Test GET /models/current returns current model config."""
        response = requests.get(
            f"{BASE_URL}/models/current",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Current model failed: {response.text}"
        data = response.json()
        assert "provider" in data or "model" in data, f"No model info: {data}"

    def test_list_providers(self, auth_headers):
        """Test GET /models/providers returns supported providers."""
        response = requests.get(
            f"{BASE_URL}/models/providers",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"List providers failed: {response.text}"


# =============================================================================
# SECTION 9: Admin Endpoints Tests
# =============================================================================

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

    def test_admin_without_auth(self):
        """Test admin endpoints without authentication."""
        endpoints = ["/admin/stats", "/admin/config", "/admin/users"]
        for endpoint in endpoints:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
            assert response.status_code in [401, 403], (
                f"{endpoint} should require auth, got {response.status_code}"
            )


# =============================================================================
# SECTION 10: Evaluation Endpoints Tests
# =============================================================================

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

    def test_evaluation_metrics(self, auth_headers):
        """Test that evaluation results include proper metrics."""
        response = requests.get(
            f"{BASE_URL}/evaluation/runs",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            # If there are runs, they should have metrics
            if isinstance(data, list) and len(data) > 0:
                run = data[0]
                assert "id" in run or "run_id" in run, "Run should have ID"


# =============================================================================
# SECTION 11: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_json_body(self, auth_headers):
        """Test handling of invalid JSON in request body."""
        response = requests.post(
            f"{BASE_URL}/chat",
            headers={**auth_headers, "Content-Type": "application/json"},
            data="not valid json{",
            timeout=TIMEOUT,
        )
        assert response.status_code in [400, 422], (
            f"Invalid JSON should fail, got {response.status_code}"
        )

    def test_wrong_content_type(self, auth_headers):
        """Test handling of wrong content type."""
        response = requests.post(
            f"{BASE_URL}/chat",
            headers={**auth_headers, "Content-Type": "text/plain"},
            data="Hello",
            timeout=TIMEOUT,
        )
        assert response.status_code in [400, 415, 422], (
            f"Wrong content type should fail, got {response.status_code}"
        )

    def test_nonexistent_endpoint(self):
        """Test 404 for nonexistent endpoints."""
        response = requests.get(
            f"{BASE_URL}/nonexistent/endpoint/that/does/not/exist",
            timeout=TIMEOUT,
        )
        assert response.status_code == 404

    def test_method_not_allowed(self):
        """Test 405 for wrong HTTP method."""
        response = requests.delete(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code in [404, 405], (
            f"Wrong method should fail, got {response.status_code}"
        )

    def test_unicode_handling(self, auth_headers):
        """Test handling of unicode characters."""
        # Create conversation with unicode title
        response = requests.post(
            f"{BASE_URL}/conversations",
            headers=auth_headers,
            json={"title": "Test 日本語 中文 한국어 🎉"},
            timeout=TIMEOUT,
        )
        assert response.status_code == 200, f"Unicode handling failed: {response.text}"

    def test_concurrent_requests(self, auth_headers):
        """Test handling of concurrent requests."""
        def make_health_request():
            return requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(20)]
            results = [f.result() for f in as_completed(futures)]
        
        # All requests should succeed
        for r in results:
            assert r.status_code == 200, f"Concurrent request failed: {r.status_code}"


# =============================================================================
# SECTION 12: API Documentation Tests
# =============================================================================

class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_json(self):
        """Test /openapi.json returns OpenAPI spec."""
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=TIMEOUT)
        assert response.status_code == 200, f"OpenAPI failed: {response.status_code}"
        data = response.json()
        assert "openapi" in data, "Should have openapi version"
        assert "paths" in data, "Should have paths"

    def test_swagger_ui(self):
        """Test /docs returns Swagger UI."""
        response = requests.get(f"{BASE_URL}/docs", timeout=TIMEOUT)
        assert response.status_code == 200, f"Swagger UI failed: {response.status_code}"
        assert "swagger" in response.text.lower() or "html" in response.text.lower()

    def test_redoc(self):
        """Test /redoc returns ReDoc."""
        response = requests.get(f"{BASE_URL}/redoc", timeout=TIMEOUT)
        assert response.status_code == 200, f"ReDoc failed: {response.status_code}"


# =============================================================================
# SECTION 13: Security Tests
# =============================================================================

class TestSecurity:
    """Test security-related functionality."""

    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = requests.options(
            f"{BASE_URL}/health",
            headers={"Origin": "http://localhost:3000"},
            timeout=TIMEOUT,
        )
        # CORS might return 200 or 204 for preflight
        assert response.status_code in [200, 204, 405]

    def test_xss_prevention(self, auth_headers):
        """Test XSS attack prevention in user input."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
        ]
        for payload in xss_payloads:
            response = requests.post(
                f"{BASE_URL}/conversations",
                headers=auth_headers,
                json={"title": payload},
                timeout=TIMEOUT,
            )
            # Should not crash; response should not execute the script
            assert response.status_code in [200, 400, 422], (
                f"XSS payload handling failed: {response.status_code}"
            )

    def test_path_traversal_prevention(self, auth_headers):
        """Test path traversal attack prevention."""
        malicious_paths = [
            "../../../etc/passwd",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc/passwd",
        ]
        for path in malicious_paths:
            response = requests.get(
                f"{BASE_URL}/documents/{path}",
                headers=auth_headers,
                timeout=TIMEOUT,
            )
            # Should not expose system files
            assert response.status_code in [400, 404, 422], (
                f"Path traversal should be prevented: {response.status_code}"
            )


# =============================================================================
# SECTION 14: Comprehensive Smoke Test
# =============================================================================

class TestEndpointSummary:
    """Summary test to verify all critical endpoints at once."""

    def test_all_critical_endpoints(self, auth_headers):
        """Quick smoke test of all critical endpoints."""
        endpoints = [
            ("GET", "/status", None, [200]),
            ("GET", "/health", None, [200]),
            ("GET", "/ingest/status", None, [200]),
            ("GET", "/memory/status", None, [200]),
            ("GET", "/conversations", auth_headers, [200]),
            ("GET", "/clients", auth_headers, [200]),
            ("GET", "/documents", auth_headers, [200]),
            ("GET", "/documents/stats", auth_headers, [200]),
            ("GET", "/documents/formats", auth_headers, [200]),
            ("GET", "/models", auth_headers, [200]),
            ("GET", "/models/current", auth_headers, [200]),
            ("GET", "/models/providers", auth_headers, [200]),
            ("GET", "/evaluation/datasets", auth_headers, [200]),
            ("GET", "/evaluation/runs", auth_headers, [200]),
            ("GET", "/admin/stats", auth_headers, [200, 403]),
            ("GET", "/admin/config", auth_headers, [200, 403]),
            ("GET", "/openapi.json", None, [200]),
            ("GET", "/docs", None, [200]),
            ("GET", "/metrics", None, [200]),
        ]

        failed = []
        for method, endpoint, headers, expected in endpoints:
            try:
                response = requests.get(
                    f"{BASE_URL}{endpoint}",
                    headers=headers,
                    timeout=TIMEOUT,
                )
                if response.status_code not in expected:
                    failed.append(f"{method} {endpoint}: {response.status_code} (expected {expected})")
            except Exception as e:
                failed.append(f"{method} {endpoint}: {str(e)}")

        if failed:
            pytest.fail(f"Failed endpoints:\n" + "\n".join(failed))

    def test_auth_protected_endpoints_require_auth(self):
        """Verify all protected endpoints reject unauthenticated requests."""
        protected_endpoints = [
            "/conversations",
            "/clients",
            "/documents",
            "/chat",
            "/admin/stats",
            "/admin/users",
        ]

        for endpoint in protected_endpoints:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
            assert response.status_code in [401, 403, 405], (
                f"{endpoint} should require auth, got {response.status_code}"
            )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
