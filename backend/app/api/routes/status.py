from fastapi import APIRouter

from app.rag.vector_store import get_vector_store

router = APIRouter()


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

