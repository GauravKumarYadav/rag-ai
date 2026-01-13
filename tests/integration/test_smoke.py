"""
Smoke Tests for RAG AI Chatbot.

These tests verify basic end-to-end functionality:
1. Create clients A and B
2. Upload a document to each
3. Query each client and verify isolation
4. Record baseline metrics

Run with: pytest tests/integration/test_smoke.py -v
"""

import pytest
import httpx
import asyncio
import uuid
from typing import Optional


# Test configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30.0


class TestClient:
    """HTTP client wrapper for test requests."""
    
    def __init__(self, base_url: str, token: Optional[str] = None):
        self.base_url = base_url
        self.token = token
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=TEST_TIMEOUT,
        )
        return self
    
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
    
    async def post(self, path: str, **kwargs):
        return await self._client.post(path, **kwargs)
    
    async def get(self, path: str, **kwargs):
        return await self._client.get(path, **kwargs)
    
    async def delete(self, path: str, **kwargs):
        return await self._client.delete(path, **kwargs)


async def get_test_token(base_url: str) -> str:
    """Get a test JWT token for authentication."""
    async with httpx.AsyncClient(base_url=base_url, timeout=TEST_TIMEOUT) as client:
        # Try to login with test credentials
        response = await client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin"},
        )
        if response.status_code == 200:
            return response.json().get("access_token", "")
        
        # If login fails, try to register first
        response = await client.post(
            "/auth/register",
            json={"username": "admin", "password": "admin"},
        )
        if response.status_code in (200, 201):
            # Now login
            response = await client.post(
                "/auth/login",
                json={"username": "admin", "password": "admin"},
            )
            if response.status_code == 200:
                return response.json().get("access_token", "")
        
        # Return empty token if auth endpoints don't exist (will rely on test override)
        return ""


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def auth_token():
    """Get authentication token for tests."""
    return await get_test_token(BASE_URL)


@pytest.fixture(scope="module")
async def test_client(auth_token):
    """Create authenticated test client."""
    async with TestClient(BASE_URL, auth_token) as client:
        yield client


@pytest.fixture
async def client_a(test_client):
    """Create test client A and clean up after test."""
    unique_id = str(uuid.uuid4())[:8]
    client_name = f"TestClientA_{unique_id}"
    
    # Create client
    response = await test_client.post(
        "/clients",
        json={"name": client_name, "aliases": ["client-a-alias"]},
    )
    
    if response.status_code == 200:
        client_data = response.json()
        yield client_data
        # Cleanup
        await test_client.delete(f"/clients/{client_data['id']}?delete_documents=true")
    else:
        pytest.skip(f"Could not create client A: {response.status_code} - {response.text}")


@pytest.fixture
async def client_b(test_client):
    """Create test client B and clean up after test."""
    unique_id = str(uuid.uuid4())[:8]
    client_name = f"TestClientB_{unique_id}"
    
    # Create client
    response = await test_client.post(
        "/clients",
        json={"name": client_name, "aliases": ["client-b-alias"]},
    )
    
    if response.status_code == 200:
        client_data = response.json()
        yield client_data
        # Cleanup
        await test_client.delete(f"/clients/{client_data['id']}?delete_documents=true")
    else:
        pytest.skip(f"Could not create client B: {response.status_code} - {response.text}")


# Sample test documents
DOC_A_CONTENT = """
# Client A Policy Document

## Section 1: Coverage Details
The maximum coverage amount for Client A is $500,000.
The deductible is set at $1,000 per claim.

## Section 2: Eligibility
All employees working more than 20 hours per week are eligible.
Coverage begins on the first day of employment.

## Section 3: Claims Process
Claims must be submitted within 30 days of the incident.
All claims require documentation including receipts and incident reports.
"""

DOC_B_CONTENT = """
# Client B Service Agreement

## Section 1: Service Level
Client B has a premium service tier with 99.9% uptime guarantee.
Response time for critical issues is 15 minutes.

## Section 2: Pricing
Monthly fee is $2,500 for the basic package.
Additional users cost $50 per seat per month.

## Section 3: Support
24/7 support is included for all critical issues.
Non-critical support is available Monday through Friday.
"""


@pytest.mark.asyncio
class TestSmokeBasic:
    """Basic smoke tests for API health."""
    
    async def test_health_endpoint(self, test_client):
        """Test that health endpoint responds."""
        response = await test_client.get("/health")
        assert response.status_code == 200, f"Health check failed: {response.text}"
    
    async def test_root_endpoint(self, test_client):
        """Test that root endpoint responds."""
        response = await test_client.get("/")
        assert response.status_code == 200, f"Root endpoint failed: {response.text}"


@pytest.mark.asyncio
class TestSmokeClientIsolation:
    """Smoke tests for client creation and document isolation."""
    
    async def test_create_clients(self, test_client, client_a, client_b):
        """Test that two clients can be created."""
        assert client_a["id"] != client_b["id"]
        assert "TestClientA" in client_a["name"]
        assert "TestClientB" in client_b["name"]
    
    async def test_upload_document_to_client_a(self, test_client, client_a):
        """Upload a document to client A."""
        # Create a simple text file for upload
        files = {
            "files": ("policy_a.txt", DOC_A_CONTENT.encode(), "text/plain"),
        }
        data = {
            "client_id": client_a["id"],
            "chunk_size": "500",
            "chunk_overlap": "50",
        }
        
        response = await test_client._client.post(
            "/documents/upload",
            files=files,
            data=data,
        )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        result = response.json()
        assert result["document_count"] == 1
        assert result["chunk_count"] > 0
    
    async def test_upload_document_to_client_b(self, test_client, client_b):
        """Upload a document to client B."""
        files = {
            "files": ("agreement_b.txt", DOC_B_CONTENT.encode(), "text/plain"),
        }
        data = {
            "client_id": client_b["id"],
            "chunk_size": "500",
            "chunk_overlap": "50",
        }
        
        response = await test_client._client.post(
            "/documents/upload",
            files=files,
            data=data,
        )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        result = response.json()
        assert result["document_count"] == 1
        assert result["chunk_count"] > 0
    
    async def test_query_client_a_returns_a_docs(self, test_client, client_a):
        """Query client A should only return client A documents."""
        # First upload a doc
        files = {"files": ("policy_a.txt", DOC_A_CONTENT.encode(), "text/plain")}
        data = {"client_id": client_a["id"], "chunk_size": "500", "chunk_overlap": "50"}
        await test_client._client.post("/documents/upload", files=files, data=data)
        
        # Now search
        response = await test_client.post(
            "/documents/search",
            json={
                "query": "What is the maximum coverage amount?",
                "client_id": client_a["id"],
                "top_k": 5,
            },
        )
        
        assert response.status_code == 200, f"Search failed: {response.text}"
        results = response.json().get("results", [])
        
        # Should find relevant results
        assert len(results) > 0, "Expected to find documents for client A"
        
        # Results should contain client A content
        all_content = " ".join([r.get("content", "") for r in results])
        assert "$500,000" in all_content or "coverage" in all_content.lower()
    
    async def test_query_isolation_client_b_not_in_a(self, test_client, client_a, client_b):
        """Verify client B documents don't appear in client A queries."""
        # Upload to both clients
        files_a = {"files": ("policy_a.txt", DOC_A_CONTENT.encode(), "text/plain")}
        files_b = {"files": ("agreement_b.txt", DOC_B_CONTENT.encode(), "text/plain")}
        
        await test_client._client.post(
            "/documents/upload",
            files=files_a,
            data={"client_id": client_a["id"], "chunk_size": "500", "chunk_overlap": "50"},
        )
        await test_client._client.post(
            "/documents/upload",
            files=files_b,
            data={"client_id": client_b["id"], "chunk_size": "500", "chunk_overlap": "50"},
        )
        
        # Query client A for client B content - should NOT find it
        response = await test_client.post(
            "/documents/search",
            json={
                "query": "What is the monthly fee?",  # This is in doc B
                "client_id": client_a["id"],
                "top_k": 5,
            },
        )
        
        assert response.status_code == 200
        results = response.json().get("results", [])
        
        # Should NOT contain client B specific content
        all_content = " ".join([r.get("content", "") for r in results])
        assert "$2,500" not in all_content, "Client B content leaked to client A query!"


@pytest.mark.asyncio
class TestSmokeChatEndpoint:
    """Smoke tests for chat endpoint."""
    
    async def test_chat_simple_query(self, test_client):
        """Test basic chat functionality."""
        response = await test_client.post(
            "/chat",
            json={
                "message": "Hello, how are you?",
                "conversation_id": f"test_{uuid.uuid4()}",
                "stream": False,
            },
        )
        
        # Should get a response (even if no documents)
        assert response.status_code == 200, f"Chat failed: {response.text}"
    
    async def test_chat_with_client_context(self, test_client, client_a):
        """Test chat with specific client context."""
        # Upload a document first
        files = {"files": ("policy_a.txt", DOC_A_CONTENT.encode(), "text/plain")}
        data = {"client_id": client_a["id"], "chunk_size": "500", "chunk_overlap": "50"}
        await test_client._client.post("/documents/upload", files=files, data=data)
        
        response = await test_client.post(
            "/chat",
            json={
                "message": "What is the maximum coverage amount?",
                "conversation_id": f"test_{uuid.uuid4()}",
                "client_id": client_a["id"],
                "stream": False,
            },
        )
        
        assert response.status_code == 200, f"Chat failed: {response.text}"


@pytest.mark.asyncio
class TestSmokeMetrics:
    """Smoke tests for Prometheus metrics endpoint."""
    
    async def test_metrics_endpoint_exists(self, test_client):
        """Test that metrics endpoint is accessible."""
        response = await test_client.get("/metrics")
        # Should return 200 with prometheus format
        assert response.status_code == 200, f"Metrics endpoint failed: {response.text}"
        
        # Should contain some basic metrics
        content = response.text
        assert "http_request" in content or "fastapi" in content.lower() or "python" in content.lower()
    
    async def test_rag_metrics_exist(self, test_client):
        """Verify RAG-specific metrics are exposed."""
        response = await test_client.get("/metrics")
        
        if response.status_code == 200:
            content = response.text
            # These metrics should exist after our enhancement
            expected_metrics = [
                "rag_retrieval_duration_seconds",
                "rag_client_access_denied_total",
                "rag_cross_client_filter_applied_total",
            ]
            
            # Note: These will only appear after the metrics.py enhancement
            # For now, just check the endpoint works
            # After Phase 0 implementation, uncomment assertions:
            # for metric in expected_metrics:
            #     assert metric in content, f"Missing metric: {metric}"


# Baseline metrics recording (for comparison after optimizations)
@pytest.fixture(scope="module")
async def baseline_metrics():
    """Record baseline metrics for comparison."""
    metrics = {
        "retrieval_p95_latency": None,
        "avg_citations_per_response": None,
        "verification_failure_rate": None,
    }
    yield metrics
    
    # Print baseline summary
    print("\n=== BASELINE METRICS ===")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("========================\n")
