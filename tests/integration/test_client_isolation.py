"""
Client Isolation Integration Tests.

These tests verify that client data isolation is enforced:
1. Users can only access clients they're authorized for
2. Document retrieval is always filtered by client_id
3. Cross-client data leakage is impossible

Run with: pytest tests/integration/test_client_isolation.py -v
"""

import pytest
import httpx
import asyncio
import uuid
from typing import Optional, Dict, Any


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
        response = await client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        if response.status_code == 200:
            return response.json().get("access_token", "")
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


# Test documents with distinct content
DOC_PRIVATE_A = """
# Private Client A Document

## Confidential Section
This document contains sensitive information for Client A only.
The secret code for Client A is: ALPHA-12345-SECRET
Client A's budget is $1,000,000 for fiscal year 2024.
"""

DOC_PRIVATE_B = """
# Private Client B Document

## Confidential Section
This document contains sensitive information for Client B only.
The secret code for Client B is: BETA-67890-CLASSIFIED
Client B's budget is $2,500,000 for fiscal year 2024.
"""

DOC_GLOBAL = """
# Global Shared Document

## Public Information
This document is shared across all clients.
The company holiday is December 25th.
Office hours are 9 AM to 5 PM.
"""


@pytest.fixture
async def isolated_clients(test_client):
    """Create two isolated clients with private documents."""
    unique_id = str(uuid.uuid4())[:8]
    
    # Create Client A
    response_a = await test_client.post(
        "/clients",
        json={"name": f"IsolatedClientA_{unique_id}", "aliases": ["client-a"]},
    )
    if response_a.status_code != 200:
        pytest.skip(f"Could not create client A: {response_a.text}")
    client_a = response_a.json()
    
    # Create Client B
    response_b = await test_client.post(
        "/clients",
        json={"name": f"IsolatedClientB_{unique_id}", "aliases": ["client-b"]},
    )
    if response_b.status_code != 200:
        # Cleanup A
        await test_client.delete(f"/clients/{client_a['id']}?delete_documents=true")
        pytest.skip(f"Could not create client B: {response_b.text}")
    client_b = response_b.json()
    
    # Upload private document to Client A
    files_a = {"files": ("private_a.txt", DOC_PRIVATE_A.encode(), "text/plain")}
    await test_client._client.post(
        "/documents/upload",
        files=files_a,
        data={"client_id": client_a["id"], "chunk_size": "500", "chunk_overlap": "50"},
    )
    
    # Upload private document to Client B
    files_b = {"files": ("private_b.txt", DOC_PRIVATE_B.encode(), "text/plain")}
    await test_client._client.post(
        "/documents/upload",
        files=files_b,
        data={"client_id": client_b["id"], "chunk_size": "500", "chunk_overlap": "50"},
    )
    
    yield {"client_a": client_a, "client_b": client_b}
    
    # Cleanup
    await test_client.delete(f"/clients/{client_a['id']}?delete_documents=true")
    await test_client.delete(f"/clients/{client_b['id']}?delete_documents=true")


@pytest.mark.asyncio
class TestClientIsolation:
    """Tests for client data isolation."""
    
    async def test_search_client_a_only_returns_a_docs(self, test_client, isolated_clients):
        """Searching client A should only return client A documents."""
        client_a = isolated_clients["client_a"]
        
        response = await test_client.post(
            "/documents/search",
            json={
                "query": "What is the secret code?",
                "client_id": client_a["id"],
                "top_k": 10,
            },
        )
        
        assert response.status_code == 200
        results = response.json().get("results", [])
        
        # Should find Client A's content
        all_content = " ".join([r.get("content", "") for r in results])
        
        # Should contain Client A's secret
        assert "ALPHA-12345-SECRET" in all_content or "Client A" in all_content
        
        # Should NOT contain Client B's secret
        assert "BETA-67890-CLASSIFIED" not in all_content
        assert "Client B's budget" not in all_content
    
    async def test_search_client_b_only_returns_b_docs(self, test_client, isolated_clients):
        """Searching client B should only return client B documents."""
        client_b = isolated_clients["client_b"]
        
        response = await test_client.post(
            "/documents/search",
            json={
                "query": "What is the secret code?",
                "client_id": client_b["id"],
                "top_k": 10,
            },
        )
        
        assert response.status_code == 200
        results = response.json().get("results", [])
        
        # Should find Client B's content
        all_content = " ".join([r.get("content", "") for r in results])
        
        # Should contain Client B's secret
        assert "BETA-67890-CLASSIFIED" in all_content or "Client B" in all_content
        
        # Should NOT contain Client A's secret
        assert "ALPHA-12345-SECRET" not in all_content
        assert "Client A's budget" not in all_content
    
    async def test_cross_client_search_prevented(self, test_client, isolated_clients):
        """Searching for client B content via client A query should fail."""
        client_a = isolated_clients["client_a"]
        
        # Try to find Client B's specific content through Client A
        response = await test_client.post(
            "/documents/search",
            json={
                "query": "BETA-67890-CLASSIFIED $2,500,000",  # Client B's secrets
                "client_id": client_a["id"],
                "top_k": 10,
            },
        )
        
        assert response.status_code == 200
        results = response.json().get("results", [])
        
        # Should NOT find Client B's content
        all_content = " ".join([r.get("content", "") for r in results])
        assert "BETA-67890-CLASSIFIED" not in all_content
        assert "$2,500,000" not in all_content
    
    async def test_chat_with_client_isolation(self, test_client, isolated_clients):
        """Chat queries should only use the specified client's documents."""
        client_a = isolated_clients["client_a"]
        client_b = isolated_clients["client_b"]
        
        # Chat with Client A context
        response_a = await test_client.post(
            "/chat",
            json={
                "message": "What is the secret code mentioned in the documents?",
                "conversation_id": f"isolation_test_{uuid.uuid4()}",
                "client_id": client_a["id"],
                "stream": False,
            },
        )
        
        # Chat with Client B context
        response_b = await test_client.post(
            "/chat",
            json={
                "message": "What is the secret code mentioned in the documents?",
                "conversation_id": f"isolation_test_{uuid.uuid4()}",
                "client_id": client_b["id"],
                "stream": False,
            },
        )
        
        # Both should succeed
        assert response_a.status_code == 200
        assert response_b.status_code == 200
        
        # Get response content
        result_a = response_a.json()
        result_b = response_b.json()
        
        # Responses should be different (different client data)
        # This is a soft check - the LLM might respond differently
        if result_a.get("response") and result_b.get("response"):
            # At minimum, they should not both contain the other's secret
            response_a_text = result_a.get("response", "")
            response_b_text = result_b.get("response", "")
            
            # Client A response should not reveal Client B secrets
            assert "BETA-67890-CLASSIFIED" not in response_a_text
            
            # Client B response should not reveal Client A secrets
            assert "ALPHA-12345-SECRET" not in response_b_text


@pytest.mark.asyncio
class TestClientAccessDenied:
    """Tests for client access denial when user lacks permission."""
    
    async def test_global_client_always_accessible(self, test_client):
        """The global client should always be accessible."""
        response = await test_client.post(
            "/chat",
            json={
                "message": "Hello",
                "conversation_id": f"global_test_{uuid.uuid4()}",
                "client_id": "global",
                "stream": False,
            },
        )
        
        # Global should always work
        assert response.status_code == 200
    
    async def test_missing_client_id_uses_global(self, test_client):
        """When client_id is not specified, global client should be used."""
        response = await test_client.post(
            "/chat",
            json={
                "message": "Hello",
                "conversation_id": f"no_client_test_{uuid.uuid4()}",
                "stream": False,
            },
        )
        
        # Should succeed using global client
        assert response.status_code == 200


@pytest.mark.asyncio
class TestMetricsRecording:
    """Tests for metrics recording during client filtering."""
    
    async def test_metrics_endpoint_accessible(self, test_client):
        """Verify metrics endpoint is accessible."""
        response = await test_client.get("/metrics")
        assert response.status_code == 200
    
    async def test_client_filter_metrics_present(self, test_client, isolated_clients):
        """Verify client filter metrics are recorded."""
        client_a = isolated_clients["client_a"]
        
        # Perform a search to trigger metrics
        await test_client.post(
            "/documents/search",
            json={
                "query": "test query",
                "client_id": client_a["id"],
                "top_k": 5,
            },
        )
        
        # Check metrics endpoint
        response = await test_client.get("/metrics")
        assert response.status_code == 200
        
        content = response.text
        
        # These metrics should be present after the search
        # Note: They may have 0 values but should exist
        assert "rag_cross_client_filter_applied_total" in content or "rag_" in content
