"""
Admin API routes for system configuration and basic stats.

Simplified admin endpoints without MySQL dependency.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.auth.dependencies import require_superuser
from app.auth.users import list_all_users, create_user, update_user, get_user_by_id
from app.config import settings
from app.core.logging import get_logger


router = APIRouter()
logger = get_logger(__name__)


class SystemConfigResponse(BaseModel):
    """System configuration response."""
    llm_provider: str
    llm_model: str
    embedding_model: str
    rag_provider: str
    agent_enabled: bool
    bm25_enabled: bool
    reranker_enabled: bool
    log_level: str


@router.get("/config", response_model=SystemConfigResponse)
async def get_config(current_user: dict = Depends(require_superuser)):
    """
    Get system configuration.
    
    Returns current configuration settings for admin visibility.
    """
    return SystemConfigResponse(
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.lmstudio.model,
        embedding_model=settings.rag.embedding_model,
        rag_provider=settings.rag.provider,
        agent_enabled=settings.agent.enabled,
        bm25_enabled=settings.rag.bm25_enabled,
        reranker_enabled=settings.rag.reranker_enabled,
        log_level=settings.logging.level,
    )


@router.get("/health")
async def admin_health(current_user: dict = Depends(require_superuser)):
    """
    Get detailed health status for admin.
    
    Returns health of all system components.
    """
    health_status = {
        "status": "healthy",
        "components": {}
    }
    
    # Check Redis
    try:
        from app.memory.session_buffer import get_session_buffer
        buffer = get_session_buffer()
        if buffer._redis:
            buffer._redis.ping()
            health_status["components"]["redis"] = {"status": "healthy"}
        else:
            health_status["components"]["redis"] = {"status": "degraded", "message": "Using in-memory storage"}
    except Exception as e:
        health_status["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check ChromaDB
    try:
        from app.rag.factory import get_vector_store
        store = get_vector_store()
        if store.health_check():
            stats = store.get_stats()
            health_status["components"]["chromadb"] = {
                "status": "healthy",
                "document_count": stats.get("document_count", 0)
            }
        else:
            health_status["components"]["chromadb"] = {"status": "unhealthy"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["chromadb"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check LM Studio
    try:
        from app.clients.lmstudio import get_lmstudio_client
        client = get_lmstudio_client()
        if await client.healthcheck():
            health_status["components"]["lmstudio"] = {"status": "healthy"}
        else:
            health_status["components"]["lmstudio"] = {"status": "unhealthy"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["lmstudio"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    return health_status


@router.get("/stats")
async def get_stats(current_user: dict = Depends(require_superuser)):
    """
    Get system statistics.
    
    Returns stats about documents, sessions, and memory usage.
    """
    stats = {
        "documents": {},
        "sessions": {},
        "memory": {},
    }
    
    # Document stats from ChromaDB
    try:
        from app.rag.factory import get_vector_store
        store = get_vector_store()
        store_stats = store.get_stats()
        stats["documents"] = {
            "total_chunks": store_stats.get("document_count", 0),
            "memories": store_stats.get("memory_count", 0),
        }
    except Exception as e:
        stats["documents"]["error"] = str(e)
    
    # Session stats from Redis
    try:
        from app.memory.session_buffer import get_session_buffer
        buffer = get_session_buffer()
        conversations = buffer.list_conversations()
        stats["sessions"] = {
            "active_conversations": len(conversations),
        }
    except Exception as e:
        stats["sessions"]["error"] = str(e)
    
    # Memory settings
    stats["memory"] = {
        "max_context_tokens": settings.memory.max_context_tokens,
        "summary_target_tokens": settings.memory.summary_target_tokens,
        "sliding_window_size": settings.memory.sliding_window_size,
    }
    
    return stats


# ============================================================
# User Management
# ============================================================

class UserResponse(BaseModel):
    """User response model."""
    id: str
    username: str
    email: Optional[str] = None
    is_superuser: bool
    is_active: bool


class UserListResponse(BaseModel):
    """User list response model."""
    users: List[UserResponse]
    total: int


class CreateUserRequest(BaseModel):
    """Create user request model."""
    username: str
    email: Optional[str] = None
    password: str
    is_superuser: bool = False


class UpdateUserRequest(BaseModel):
    """Update user request model."""
    email: Optional[str] = None
    password: Optional[str] = None
    is_superuser: Optional[bool] = None
    is_active: Optional[bool] = None


@router.get("/users", response_model=UserListResponse)
async def list_users(
    limit: int = Query(100, description="Maximum number of users to return"),
    current_user: dict = Depends(require_superuser),
):
    """
    List all users.
    
    Requires admin privileges.
    """
    users = await list_all_users(limit=limit)
    
    return UserListResponse(
        users=[
            UserResponse(
                id=u["id"],
                username=u["username"],
                email=u.get("email"),
                is_superuser=u.get("is_superuser", False),
                is_active=u.get("is_active", True),
            )
            for u in users
        ],
        total=len(users),
    )


@router.post("/users", response_model=UserResponse)
async def create_new_user(
    request: CreateUserRequest,
    current_user: dict = Depends(require_superuser),
):
    """
    Create a new user.
    
    Requires admin privileges.
    """
    # Validate password
    if len(request.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters",
        )
    
    user = await create_user(
        username=request.username,
        password=request.password,
        email=request.email,
        is_superuser=request.is_superuser,
    )
    
    if user is None:
        raise HTTPException(
            status_code=400,
            detail=f"Username '{request.username}' already exists",
        )
    
    return UserResponse(
        id=user["id"],
        username=user["username"],
        email=user.get("email"),
        is_superuser=user.get("is_superuser", False),
        is_active=user.get("is_active", True),
    )


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_existing_user(
    user_id: str,
    request: UpdateUserRequest,
    current_user: dict = Depends(require_superuser),
):
    """
    Update an existing user.
    
    Requires admin privileges.
    """
    # Check if user exists
    existing = await get_user_by_id(user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Validate password if provided
    if request.password and len(request.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters",
        )
    
    # Build update dict
    updates = {}
    if request.email is not None:
        updates["email"] = request.email
    if request.password is not None:
        updates["password"] = request.password
    if request.is_superuser is not None:
        updates["is_superuser"] = request.is_superuser
    if request.is_active is not None:
        updates["is_active"] = request.is_active
    
    user = await update_user(user_id, **updates)
    
    if user is None:
        raise HTTPException(status_code=500, detail="Failed to update user")
    
    return UserResponse(
        id=user["id"],
        username=user["username"],
        email=user.get("email"),
        is_superuser=user.get("is_superuser", False),
        is_active=user.get("is_active", True),
    )


# ============================================================
# Audit Logs
# ============================================================

import json
from datetime import datetime
import redis

AUDIT_LOG_KEY = "audit:logs"
AUDIT_ACTIONS_KEY = "audit:actions"


def _get_audit_redis() -> Optional[redis.Redis]:
    """Get Redis client for audit logs."""
    try:
        r = redis.from_url(settings.redis.url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


class AuditLogEntry(BaseModel):
    """Audit log entry model."""
    id: str
    timestamp: str
    username: str
    action: str
    method: str
    path: str
    status_code: int
    details: Optional[str] = None


class AuditLogsResponse(BaseModel):
    """Audit logs response model."""
    logs: List[AuditLogEntry]
    total: int
    page: int
    page_size: int


@router.get("/audit-logs/actions")
async def get_audit_log_actions(
    current_user: dict = Depends(require_superuser),
):
    """
    Get available audit log action types.
    
    Requires admin privileges.
    """
    # Return common action types
    actions = [
        "login",
        "logout",
        "create_user",
        "update_user",
        "delete_user",
        "create_client",
        "update_client",
        "delete_client",
        "upload_document",
        "delete_document",
        "chat",
        "grant_access",
        "revoke_access",
    ]
    
    return {"actions": actions}


@router.get("/audit-logs", response_model=AuditLogsResponse)
async def get_audit_logs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    username: Optional[str] = Query(None, description="Filter by username"),
    action: Optional[str] = Query(None, description="Filter by action"),
    method: Optional[str] = Query(None, description="Filter by HTTP method"),
    status_code: Optional[int] = Query(None, description="Filter by status code"),
    current_user: dict = Depends(require_superuser),
):
    """
    Get audit logs with pagination and filtering.
    
    Requires admin privileges.
    """
    r = _get_audit_redis()
    
    if r is None:
        # Return empty if Redis not available
        return AuditLogsResponse(
            logs=[],
            total=0,
            page=page,
            page_size=page_size,
        )
    
    # Get all logs (stored as list in Redis)
    raw_logs = r.lrange(AUDIT_LOG_KEY, 0, -1)
    
    logs = []
    for raw in raw_logs:
        try:
            log = json.loads(raw)
            
            # Apply filters
            if username and log.get("username") != username:
                continue
            if action and log.get("action") != action:
                continue
            if method and log.get("method") != method:
                continue
            if status_code and log.get("status_code") != status_code:
                continue
            
            logs.append(AuditLogEntry(**log))
        except Exception:
            continue
    
    # Sort by timestamp descending (newest first)
    logs.sort(key=lambda x: x.timestamp, reverse=True)
    
    # Paginate
    total = len(logs)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = logs[start:end]
    
    return AuditLogsResponse(
        logs=paginated,
        total=total,
        page=page,
        page_size=page_size,
    )


async def log_audit_event(
    username: str,
    action: str,
    method: str,
    path: str,
    status_code: int,
    details: Optional[str] = None,
) -> None:
    """
    Log an audit event.
    
    Args:
        username: The user who performed the action
        action: The action type (e.g., 'login', 'upload_document')
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        details: Optional additional details
    """
    r = _get_audit_redis()
    if r is None:
        return
    
    import uuid
    
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "username": username,
        "action": action,
        "method": method,
        "path": path,
        "status_code": status_code,
        "details": details,
    }
    
    # Add to list (newest first)
    r.lpush(AUDIT_LOG_KEY, json.dumps(entry))
    
    # Keep only last 10000 entries
    r.ltrim(AUDIT_LOG_KEY, 0, 9999)
