"""
Conftest for live integration tests.

These tests run against actual running containers and don't need app imports.
Provides fixtures for authentication, cleanup, and test data generation.
"""
import os
import time
import uuid
import pytest
import requests
from typing import Generator, Dict, Optional, List

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = os.getenv("API_BASE_URL", "http://localhost")
TEST_USERNAME = os.getenv("TEST_USERNAME", "admin")
TEST_PASSWORD = os.getenv("TEST_PASSWORD", "admin123")
TIMEOUT = 30
LONG_TIMEOUT = 120


# =============================================================================
# Authentication Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def auth_token() -> str:
    """Get authentication token for protected endpoints.
    
    Session-scoped to avoid repeated login calls.
    """
    try:
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
            timeout=TIMEOUT,
        )
        if response.status_code != 200:
            pytest.skip(f"Could not get auth token: {response.text}")
        return response.json()["access_token"]
    except requests.ConnectionError as e:
        pytest.skip(f"Backend not accessible: {e}")


@pytest.fixture(scope="session")
def auth_headers(auth_token: str) -> Dict[str, str]:
    """Get authentication headers for protected endpoints."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture(scope="function")
def fresh_auth_token() -> str:
    """Get a fresh authentication token (function-scoped).
    
    Use this when testing token expiry or needing isolated tokens.
    """
    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        pytest.skip(f"Could not get fresh auth token: {response.text}")
    return response.json()["access_token"]


@pytest.fixture(scope="function")
def fresh_auth_headers(fresh_auth_token: str) -> Dict[str, str]:
    """Get fresh authentication headers (function-scoped)."""
    return {"Authorization": f"Bearer {fresh_auth_token}"}


# =============================================================================
# URL Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def base_url() -> str:
    """Get base URL for API calls."""
    return BASE_URL


@pytest.fixture(scope="session")
def ollama_url() -> str:
    """Get Ollama service URL."""
    return os.getenv("OLLAMA_URL", "http://localhost:11434")


@pytest.fixture(scope="session")
def chromadb_url() -> str:
    """Get ChromaDB URL."""
    return os.getenv("CHROMADB_URL", "http://localhost:8020")


# =============================================================================
# Test Data Generation Fixtures
# =============================================================================

@pytest.fixture
def unique_name() -> str:
    """Generate a unique name for test resources."""
    return f"test_{int(time.time())}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_document_content() -> bytes:
    """Generate test document content."""
    return b"""
    This is a test document for the RAG chatbot system.
    
    It contains multiple paragraphs of text that can be used
    for testing document upload, chunking, and retrieval.
    
    Key topics covered:
    - RAG (Retrieval Augmented Generation)
    - Vector databases
    - Document processing
    - Semantic search
    
    This document should be indexed and retrievable via search queries.
    """


@pytest.fixture
def large_document_content() -> bytes:
    """Generate a larger test document (~100KB)."""
    paragraph = """
    This is a paragraph of text that will be repeated to create a larger document.
    The purpose is to test the system's handling of larger files during upload
    and processing. Each paragraph contains enough text to be meaningful for
    testing chunking and retrieval operations.
    """ * 10
    return (paragraph * 100).encode()


# =============================================================================
# Resource Cleanup Fixtures
# =============================================================================

@pytest.fixture
def test_conversation(auth_headers: Dict[str, str]) -> Generator[Dict, None, None]:
    """Create a test conversation and clean it up after the test."""
    unique_title = f"test_conv_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    response = requests.post(
        f"{BASE_URL}/conversations",
        headers=auth_headers,
        json={"title": unique_title},
        timeout=TIMEOUT,
    )
    
    if response.status_code != 200:
        pytest.skip(f"Could not create test conversation: {response.text}")
    
    conversation = response.json()
    yield conversation
    
    # Cleanup
    try:
        requests.delete(
            f"{BASE_URL}/conversations/{conversation['id']}",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
    except Exception:
        pass  # Ignore cleanup failures


@pytest.fixture
def test_client(auth_headers: Dict[str, str]) -> Generator[Dict, None, None]:
    """Create a test client and clean it up after the test."""
    unique_name = f"test_client_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    response = requests.post(
        f"{BASE_URL}/clients",
        headers=auth_headers,
        json={"name": unique_name},
        timeout=TIMEOUT,
    )
    
    if response.status_code != 200:
        pytest.skip(f"Could not create test client: {response.text}")
    
    client = response.json()
    yield client
    
    # Cleanup
    try:
        requests.delete(
            f"{BASE_URL}/clients/{client['id']}",
            headers=auth_headers,
            timeout=TIMEOUT,
        )
    except Exception:
        pass  # Ignore cleanup failures


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def api_available() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_if_api_unavailable(api_available: bool, request):
    """Skip tests if API is not available (unless marked with 'allow_offline')."""
    if not api_available:
        if "allow_offline" not in [marker.name for marker in request.node.iter_markers()]:
            pytest.skip("API not available")


# =============================================================================
# Performance Tracking Fixtures
# =============================================================================

@pytest.fixture
def track_response_time():
    """Track response time for performance assertions."""
    times: List[float] = []
    
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            times.append(elapsed)
            self.elapsed = elapsed
    
    Timer.times = times
    return Timer


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take >30s)"
    )
    config.addinivalue_line(
        "markers", "llm: marks tests that require LLM interaction"
    )
    config.addinivalue_line(
        "markers", "allow_offline: marks tests that can run without API"
    )
    config.addinivalue_line(
        "markers", "security: marks security-related tests"
    )
