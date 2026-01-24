"""
Admin API routes for system configuration and basic stats.

Simplified admin endpoints without MySQL dependency.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth.dependencies import require_superuser
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
