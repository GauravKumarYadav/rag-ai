"""JWT token creation and verification utilities."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from jose import JWTError, jwt

from app.config import settings


def create_access_token(
    user_id: str,
    username: str,
    is_superuser: bool = False,
    allowed_clients: Optional[List[str]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.
    
    Args:
        user_id: The unique user identifier
        username: The username
        is_superuser: Whether the user has admin privileges
        allowed_clients: List of client IDs this user can access
        expires_delta: Custom expiration time, defaults to settings value
        
    Returns:
        Encoded JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.jwt_access_token_expire_minutes)
    
    expire = datetime.now(timezone.utc) + expires_delta
    
    to_encode: Dict[str, Any] = {
        "sub": user_id,
        "username": username,
        "is_superuser": is_superuser,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    
    # Include allowed_clients in token for fast authorization
    # Note: 'global' client is always implicitly allowed
    if allowed_clients is not None:
        to_encode["allowed_clients"] = allowed_clients
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    
    return encoded_jwt


async def create_access_token_with_clients(
    user_id: str,
    username: str,
    is_superuser: bool = False,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token with allowed_clients fetched from Redis.
    
    This is an async version that looks up the user's allowed clients
    and embeds them in the token.
    
    Args:
        user_id: The unique user identifier
        username: The username
        is_superuser: Whether the user has admin privileges
        expires_delta: Custom expiration time, defaults to settings value
        
    Returns:
        Encoded JWT token string with allowed_clients claim
    """
    from app.auth.users import get_user_client_ids
    
    # Fetch allowed clients from Redis
    allowed_clients = await get_user_client_ids(user_id)
    
    return create_access_token(
        user_id=user_id,
        username=username,
        is_superuser=is_superuser,
        allowed_clients=allowed_clients,
        expires_delta=expires_delta,
    )


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and verify a JWT token.
    
    Args:
        token: The JWT token string to decode
        
    Returns:
        Decoded token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        return None


def get_token_user_id(token: str) -> Optional[str]:
    """Extract user_id from a token without full validation."""
    payload = decode_token(token)
    if payload:
        return payload.get("sub")
    return None


def get_token_username(token: str) -> Optional[str]:
    """Extract username from a token without full validation."""
    payload = decode_token(token)
    if payload:
        return payload.get("username")
    return None
