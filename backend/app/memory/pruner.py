import asyncio
import logging
from typing import Optional

from app.rag.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

# Global scheduler task
_scheduler_task: Optional[asyncio.Task] = None


def prune_memories(store: VectorStore, max_items: int = 500) -> int:
    """
    Prune old memories to keep the collection under max_items.
    Returns the number of items deleted.
    """
    collection = store.memories
    try:
        total = collection.count()
    except Exception:
        return 0
    
    if total <= max_items:
        return 0

    records = collection.get()
    ids = records.get("ids", [])
    if not ids:
        return 0
    
    to_delete = ids[: max(0, total - max_items)]
    if to_delete:
        collection.delete(ids=to_delete)
        logger.info(f"Pruned {len(to_delete)} memories (total was {total}, max is {max_items})")
        return len(to_delete)
    return 0


def prune_documents(store: VectorStore, max_items: int = 10000) -> int:
    """
    Prune old documents to keep the collection under max_items.
    Returns the number of items deleted.
    """
    collection = store.docs
    try:
        total = collection.count()
    except Exception:
        return 0
    
    if total <= max_items:
        return 0

    records = collection.get()
    ids = records.get("ids", [])
    if not ids:
        return 0
    
    to_delete = ids[: max(0, total - max_items)]
    if to_delete:
        collection.delete(ids=to_delete)
        logger.info(f"Pruned {len(to_delete)} documents (total was {total}, max is {max_items})")
        return len(to_delete)
    return 0


async def _pruning_loop(interval_seconds: int = 3600, max_memories: int = 500, max_documents: int = 10000):
    """Background task that periodically prunes memories and documents."""
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            store = get_vector_store()
            prune_memories(store, max_items=max_memories)
            prune_documents(store, max_items=max_documents)
        except asyncio.CancelledError:
            logger.info("Pruning scheduler cancelled")
            break
        except Exception as e:
            logger.error(f"Error in pruning scheduler: {e}")


def start_pruning_scheduler(interval_seconds: int = 3600):
    """Start the background pruning scheduler."""
    global _scheduler_task
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(_pruning_loop(interval_seconds))
        logger.info(f"Started memory pruning scheduler (interval: {interval_seconds}s)")


def stop_pruning_scheduler():
    """Stop the background pruning scheduler."""
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        _scheduler_task.cancel()
        logger.info("Stopped memory pruning scheduler")

