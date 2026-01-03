"""Authentication module for JWT-based authentication."""

from app.auth.jwt import create_access_token, decode_token
from app.auth.password import hash_password, verify_password
from app.auth.dependencies import get_current_user, get_current_user_optional

__all__ = [
    "create_access_token",
    "decode_token",
    "hash_password",
    "verify_password",
    "get_current_user",
    "get_current_user_optional",
]
