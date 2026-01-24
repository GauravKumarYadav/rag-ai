"""
Simple user management with Redis persistence.

This module provides basic user authentication without MySQL.
Users are stored in Redis with a default admin user created on first access.
"""

import json
import logging
from typing import Dict, List, Optional
import uuid

import redis

from app.config import settings
from app.auth.password import hash_password, verify_password

logger = logging.getLogger(__name__)

# Redis keys
USERS_KEY = "auth:users"
USER_CLIENTS_KEY = "auth:user_clients"

# Default admin credentials
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"  # Change in production!

_redis_client: Optional[redis.Redis] = None


def _get_redis() -> Optional[redis.Redis]:
    """Get Redis client, initializing if needed."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(settings.redis.url, decode_responses=True)
            _redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed for auth: {e}")
            _redis_client = None
    return _redis_client


def _ensure_admin_user() -> None:
    """Ensure the default admin user exists."""
    r = _get_redis()
    if r is None:
        return
    
    # Check if admin user exists
    admin_data = r.hget(USERS_KEY, DEFAULT_ADMIN_USERNAME)
    if admin_data is None:
        # Create default admin user
        admin_user = {
            "id": str(uuid.uuid4()),
            "username": DEFAULT_ADMIN_USERNAME,
            "hashed_password": hash_password(DEFAULT_ADMIN_PASSWORD),
            "email": "admin@localhost",
            "is_active": True,
            "is_superuser": True,
        }
        r.hset(USERS_KEY, DEFAULT_ADMIN_USERNAME, json.dumps(admin_user))
        logger.info(f"Created default admin user: {DEFAULT_ADMIN_USERNAME}")


async def get_user_by_username(username: str) -> Optional[Dict]:
    """
    Get user by username.
    
    Args:
        username: The username to look up
        
    Returns:
        User dict or None if not found
    """
    _ensure_admin_user()
    
    r = _get_redis()
    if r is None:
        # Fallback: allow admin login without Redis
        if username == DEFAULT_ADMIN_USERNAME:
            return {
                "id": "admin-fallback",
                "username": DEFAULT_ADMIN_USERNAME,
                "hashed_password": hash_password(DEFAULT_ADMIN_PASSWORD),
                "email": "admin@localhost",
                "is_active": True,
                "is_superuser": True,
            }
        return None
    
    user_data = r.hget(USERS_KEY, username)
    if user_data:
        return json.loads(user_data)
    return None


async def get_user_client_ids(user_id: str) -> List[str]:
    """
    Get list of client IDs a user has access to.
    
    Args:
        user_id: The user's ID
        
    Returns:
        List of client IDs
    """
    r = _get_redis()
    if r is None:
        return []
    
    client_ids = r.smembers(f"{USER_CLIENTS_KEY}:{user_id}")
    return list(client_ids) if client_ids else []


async def create_user(
    username: str,
    password: str,
    email: Optional[str] = None,
    is_superuser: bool = False,
) -> Optional[Dict]:
    """
    Create a new user.
    
    Args:
        username: Unique username
        password: Plain text password (will be hashed)
        email: Optional email address
        is_superuser: Whether user is admin
        
    Returns:
        Created user dict or None if failed
    """
    r = _get_redis()
    if r is None:
        return None
    
    # Check if username exists
    if r.hexists(USERS_KEY, username):
        return None
    
    user = {
        "id": str(uuid.uuid4()),
        "username": username,
        "hashed_password": hash_password(password),
        "email": email,
        "is_active": True,
        "is_superuser": is_superuser,
    }
    
    r.hset(USERS_KEY, username, json.dumps(user))
    return user


async def add_user_client(user_id: str, client_id: str) -> bool:
    """
    Grant a user access to a client.
    
    Args:
        user_id: The user's ID
        client_id: The client ID to grant access to
        
    Returns:
        True if added, False otherwise
    """
    r = _get_redis()
    if r is None:
        return False
    
    r.sadd(f"{USER_CLIENTS_KEY}:{user_id}", client_id)
    return True
