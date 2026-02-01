"""FastAPI dependencies for authentication and authorization."""

from typing import Any, Dict, Optional, Set

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.jwt import decode_token

# HTTP Bearer token security scheme
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)

# Global client ID - everyone has access to this
GLOBAL_CLIENT_ID = "global"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Raises HTTPException 401 if token is missing or invalid.
    
    Returns:
        Dictionary containing user information from the token payload:
        - user_id: unique user ID
        - username: username
        - is_superuser: boolean
        - allowed_clients: list of client IDs (if present in token)
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
        "allowed_clients": payload.get("allowed_clients"),
    }


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional),
) -> Optional[Dict[str, Any]]:
    """
    Dependency to optionally get the current authenticated user.
    
    Returns None if no token is provided, otherwise validates the token.
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
        "allowed_clients": payload.get("allowed_clients"),
    }


async def get_allowed_clients(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Set[str]:
    """
    Dependency to get the set of client IDs the current user can access.
    
    Returns:
        Set of client IDs the user can access
    """
    # Superusers have access to everything
    if current_user.get("is_superuser"):
        from app.models.client import get_client_store
        store = get_client_store()
        all_clients = await store.list_all()
        return {c.id for c in all_clients} | {GLOBAL_CLIENT_ID}
    
    # Check JWT claims
    jwt_clients = current_user.get("allowed_clients")
    if jwt_clients is not None:
        return set(jwt_clients) | {GLOBAL_CLIENT_ID}
    
    # Fall back to getting user client IDs from Redis
    from app.auth.users import get_user_client_ids
    user_id = current_user.get("user_id")
    
    if not user_id:
        return {GLOBAL_CLIENT_ID}
    
    user_clients = await get_user_client_ids(user_id)
    return set(user_clients) | {GLOBAL_CLIENT_ID}


def require_superuser(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Dependency to require superuser/admin privileges.
    
    Raises HTTPException 403 if user is not a superuser.
    """
    if not current_user.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


async def validate_client_access(
    client_id: str,
    allowed_clients: Set[str] = Depends(get_allowed_clients),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> str:
    """
    Dependency to validate that the user has access to a specific client.
    
    Raises:
        HTTPException 403 if user does not have access to the client
        
    Returns:
        The validated client_id
    """
    if client_id not in allowed_clients:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to client '{client_id}'",
        )
    
    return client_id


def get_validated_client_id(
    client_id: Optional[str] = None,
    allowed_clients: Set[str] = Depends(get_allowed_clients),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> str:
    """
    Dependency to get a validated client ID.
    
    If client_id is provided, validates access.
    If client_id is None, returns the global client.
    """
    # Default to global client if not specified
    if client_id is None:
        return GLOBAL_CLIENT_ID
    
    # Validate access
    if client_id not in allowed_clients:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to client '{client_id}'",
        )
    
    return client_id


async def require_client_role(
    client_id: str,
    required_role: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Dependency to require a specific role for a client.
    
    For simplified auth, just checks if user has access to the client.
    """
    # Superusers bypass role checks
    if current_user.get("is_superuser"):
        return current_user
    
    # For non-superusers, check if they have access to the client
    allowed = await get_allowed_clients(current_user)
    
    if client_id not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to client '{client_id}'",
        )
    
    return current_user
