"""
Pytest configuration for unit tests.

Note: This conftest is for unit tests that require the app to be imported.
For live integration tests against running containers, use tests/live/ directory.
"""
import pytest

# Try to import app dependencies - skip if not available (e.g., for live tests)
try:
    from app.main import app
    from app.auth.dependencies import get_current_user as auth_current_user
    from app.auth.dependencies import get_current_user_optional
    from app.dependencies import get_current_user as deps_current_user
    
    APP_AVAILABLE = True
except ImportError:
    APP_AVAILABLE = False
    app = None


@pytest.fixture(autouse=True)
def override_auth_dependencies():
    """Provide a fake authenticated user for tests to bypass JWT."""
    if not APP_AVAILABLE or app is None:
        yield
        return
    
    fake_user = {"user_id": "test-user", "username": "tester", "is_superuser": True}
    app.dependency_overrides[auth_current_user] = lambda: fake_user
    app.dependency_overrides[get_current_user_optional] = lambda: fake_user
    app.dependency_overrides[deps_current_user] = lambda: fake_user
    yield
    app.dependency_overrides.clear()
