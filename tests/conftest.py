import pytest

from app.main import app
from app.auth.dependencies import get_current_user as auth_current_user
from app.auth.dependencies import get_current_user_optional
from app.dependencies import get_current_user as deps_current_user


@pytest.fixture(autouse=True)
def override_auth_dependencies():
    """Provide a fake authenticated user for tests to bypass JWT."""
    fake_user = {"user_id": "test-user", "username": "tester", "is_superuser": True}
    app.dependency_overrides[auth_current_user] = lambda: fake_user
    app.dependency_overrides[get_current_user_optional] = lambda: fake_user
    app.dependency_overrides[deps_current_user] = lambda: fake_user
    yield
    app.dependency_overrides.clear()
