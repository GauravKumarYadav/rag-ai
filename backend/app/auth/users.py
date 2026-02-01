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


async def remove_user_client(user_id: str, client_id: str) -> bool:
    """
    Revoke a user's access to a client.
    
    Args:
        user_id: The user's ID
        client_id: The client ID to revoke access from
        
    Returns:
        True if removed, False otherwise
    """
    r = _get_redis()
    if r is None:
        return False
    
    r.srem(f"{USER_CLIENTS_KEY}:{user_id}", client_id)
    return True


async def get_user_by_id(user_id: str) -> Optional[Dict]:
    """
    Get user by ID.
    
    Args:
        user_id: The user's ID
        
    Returns:
        User dict or None if not found
    """
    r = _get_redis()
    if r is None:
        return None
    
    # Scan all users to find by ID
    all_users = r.hgetall(USERS_KEY)
    for username, user_data in all_users.items():
        user = json.loads(user_data)
        if user.get("id") == user_id:
            return user
    return None


async def update_user_password(user_id: str, new_password: str) -> bool:
    """
    Update a user's password.
    
    Args:
        user_id: The user's ID
        new_password: The new plain text password (will be hashed)
        
    Returns:
        True if updated, False otherwise
    """
    r = _get_redis()
    if r is None:
        return False
    
    # Find user by ID
    all_users = r.hgetall(USERS_KEY)
    for username, user_data in all_users.items():
        user = json.loads(user_data)
        if user.get("id") == user_id:
            user["hashed_password"] = hash_password(new_password)
            r.hset(USERS_KEY, username, json.dumps(user))
            return True
    return False


async def update_user(user_id: str, **updates) -> Optional[Dict]:
    """
    Update user fields.
    
    Args:
        user_id: The user's ID
        **updates: Fields to update (email, is_superuser, is_active, password)
        
    Returns:
        Updated user dict or None if not found
    """
    r = _get_redis()
    if r is None:
        return None
    
    # Find user by ID
    all_users = r.hgetall(USERS_KEY)
    for username, user_data in all_users.items():
        user = json.loads(user_data)
        if user.get("id") == user_id:
            # Update allowed fields
            if "email" in updates and updates["email"] is not None:
                user["email"] = updates["email"]
            if "is_superuser" in updates and updates["is_superuser"] is not None:
                user["is_superuser"] = updates["is_superuser"]
            if "is_active" in updates and updates["is_active"] is not None:
                user["is_active"] = updates["is_active"]
            if "password" in updates and updates["password"] is not None:
                user["hashed_password"] = hash_password(updates["password"])
            
            r.hset(USERS_KEY, username, json.dumps(user))
            return user
    return None


async def list_all_users(limit: int = 100) -> List[Dict]:
    """
    List all users.
    
    Args:
        limit: Maximum number of users to return
        
    Returns:
        List of user dicts (without hashed_password)
    """
    _ensure_admin_user()
    
    r = _get_redis()
    if r is None:
        return []
    
    all_users = r.hgetall(USERS_KEY)
    users = []
    for username, user_data in all_users.items():
        user = json.loads(user_data)
        # Remove sensitive data
        user.pop("hashed_password", None)
        users.append(user)
        if len(users) >= limit:
            break
    
    return users


async def get_users_for_client(client_id: str) -> List[Dict]:
    """
    Get all users with access to a specific client.
    
    Args:
        client_id: The client ID
        
    Returns:
        List of user dicts with access to the client
    """
    r = _get_redis()
    if r is None:
        return []
    
    users_with_access = []
    all_users = r.hgetall(USERS_KEY)
    
    for username, user_data in all_users.items():
        user = json.loads(user_data)
        user_id = user.get("id")
        
        # Check if user has access to this client
        client_ids = r.smembers(f"{USER_CLIENTS_KEY}:{user_id}")
        if client_id in (client_ids or set()):
            user.pop("hashed_password", None)
            users_with_access.append(user)
    
    return users_with_access
