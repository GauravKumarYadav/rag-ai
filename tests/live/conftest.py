"""
Conftest for live integration tests.
These tests run against actual running containers and don't need app imports.
"""
import os
import pytest
import requests

# Configuration
BASE_URL = os.getenv("API_BASE_URL", "http://localhost")
TEST_USERNAME = os.getenv("TEST_USERNAME", "admin")
TEST_PASSWORD = os.getenv("TEST_PASSWORD", "admin123")
TIMEOUT = 30


@pytest.fixture(scope="session")
def auth_token():
    """Get authentication token for protected endpoints."""
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
def auth_headers(auth_token):
    """Get authentication headers for protected endpoints."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture(scope="session")
def base_url():
    """Get base URL for API calls."""
    return BASE_URL
