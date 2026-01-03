from fastapi import APIRouter

from app.rag.vector_store import get_vector_store
from app.config import settings

router = APIRouter()


@router.get("/status", summary="Get overall system status")
async def system_status() -> dict:
    """Get system status including model info and health."""
    store = get_vector_store()
    try:
        doc_count = store.docs.count()
        memory_count = store.memories.count()
    except Exception:
        doc_count = None
        memory_count = None
    
    return {
        "status": "ok",
        "model": settings.lmstudio_model or "LM Studio Model",
        "documents_indexed": doc_count,
        "memories_indexed": memory_count,
        "provider": "lmstudio",
    }


@router.get("/ingest/status", summary="Return document index stats")
async def ingest_status() -> dict:
    store = get_vector_store()
    try:
        total = store.docs.count()
    except Exception:
        total = None
    return {"documents_indexed": total}


@router.get("/memory/status", summary="Return memory index stats")
async def memory_status() -> dict:
    store = get_vector_store()
    try:
        total = store.memories.count()
    except Exception:
        total = None
    return {"memories_indexed": total}

