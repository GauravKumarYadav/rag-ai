"""Central Dependency Injection Module.

This module provides all FastAPI dependencies for the application.
Centralizing dependencies here makes them easy to mock for testing
and provides a single source of truth for service instantiation.

Usage:
    from app.dependencies import get_current_user, get_chat_service
    
    @router.get("/endpoint")
    async def my_endpoint(
        user: Dict = Depends(get_current_user),
        chat: ChatService = Depends(get_chat_service),
    ):
        ...
"""

from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

# HTTP Bearer token security schemes
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)


# ============================================================
# AUTHENTICATION DEPENDENCIES
# ============================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    Get the current authenticated user from JWT token.
    
    Raises HTTPException 401 if token is missing or invalid.
    
    Returns:
        Dictionary containing user information:
        - user_id: unique user identifier
        - username: user's username
        - is_superuser: boolean flag for admin users
    """
    from app.auth.jwt import decode_token
    
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
    Optionally get the current user if a token is provided.
    
    Returns None if no token is present, but validates the token if provided.
    
    Returns:
        User dictionary or None
    """
    if credentials is None:
        return None
    
    from app.auth.jwt import decode_token
    
    payload = decode_token(credentials.credentials)
    if payload is None:
        return None
    
    return {
        "user_id": payload.get("sub"),
        "username": payload.get("username"),
        "is_superuser": payload.get("is_superuser", False),
    }


def require_superuser(
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Require the current user to be a superuser.
    
    Raises HTTPException 403 if user is not a superuser.
    """
    if not user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser access required",
        )
    return user


# ============================================================
# SERVICE DEPENDENCIES
# ============================================================

def get_chat_service():
    """Get the ChatService singleton instance."""
    from app.services.chat_service import ChatService
    return ChatService()


def get_llm_client():
    """Get the current LLM client."""
    from app.clients.lmstudio import get_lmstudio_client
    return get_lmstudio_client()


def get_client_store():
    """Get the ClientStore singleton instance."""
    from app.models.client import get_client_store
    return get_client_store()


def get_client_extractor():
    """Get the ClientExtractor singleton instance."""
    from app.services.client_extractor import get_client_extractor
    return get_client_extractor()


def get_long_term_memory():
    """Get the LongTermMemory singleton instance."""
    from app.memory.long_term import get_long_term_memory
    return get_long_term_memory()


def get_vector_store():
    """Get the vector store instance based on configuration."""
    from app.rag.factory import create_vector_store
    return create_vector_store()


# ============================================================
# DATABASE DEPENDENCIES
# ============================================================

async def get_db_pool():
    """Get the MySQL connection pool."""
    from app.db.mysql import get_db_pool
    return await get_db_pool()


# ============================================================
# UTILITY DEPENDENCIES
# ============================================================

def get_settings():
    """Get the application settings."""
    return settings


def get_processor_registry():
    """Get the document processor registry."""
    from app.processors.registry import ProcessorRegistry
    return ProcessorRegistry


# ============================================================
# WEBSOCKET DEPENDENCIES
# ============================================================

def get_connection_manager():
    """Get the WebSocket connection manager."""
    from app.api.routes.websocket import ConnectionManager
    
    # Singleton pattern
    if not hasattr(get_connection_manager, "_instance"):
        get_connection_manager._instance = ConnectionManager()
    return get_connection_manager._instance
