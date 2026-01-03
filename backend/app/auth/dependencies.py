"""FastAPI dependencies for authentication."""

from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.jwt import decode_token

# HTTP Bearer token security scheme
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Raises HTTPException 401 if token is missing or invalid.
    
    Returns:
        Dictionary containing user information from the token payload:
        - sub: user_id
        - username: username
        - is_superuser: boolean
    """
    token = credentials.credentials
    
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {
        "user_id": user_id,
        "username": payload.get("username"),
        "is_superuser": payload.get("is_superuser", False),
    }


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional),
) -> Optional[Dict[str, Any]]:
    """
    Dependency to optionally get the current authenticated user.
    
    Returns None if no token is provided, otherwise validates the token.
    Useful for routes that work with or without authentication.
    
    Returns:
        User dictionary if authenticated, None otherwise
    """
    if credentials is None:
        return None
    
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload is None:
        return None
    
    user_id = payload.get("sub")
    if user_id is None:
        return None
    
    return {
        "user_id": user_id,
        "username": payload.get("username"),
        "is_superuser": payload.get("is_superuser", False),
    }


def require_superuser(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Dependency to require superuser/admin privileges.
    
    Raises HTTPException 403 if user is not a superuser.
    
    Returns:
        User dictionary if user is a superuser
    """
    if not current_user.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user
