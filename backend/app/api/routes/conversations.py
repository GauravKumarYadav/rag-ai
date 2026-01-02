from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.memory.session_buffer import get_session_buffer
from app.memory.long_term import get_long_term_memory
from app.models.schemas import ChatMessage


router = APIRouter()


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[ChatMessage]
    message_count: int


class ConversationListResponse(BaseModel):
    conversations: List[str]


class MemorySearchResponse(BaseModel):
    results: List[dict]


@router.get("/{conversation_id}", response_model=ConversationHistory, summary="Get conversation history")
async def get_conversation(conversation_id: str):
    """Get the current session history for a conversation."""
    buffer = get_session_buffer()
    messages = buffer.get(conversation_id)
    return ConversationHistory(
        conversation_id=conversation_id,
        messages=messages,
        message_count=len(messages),
    )


@router.get("", response_model=ConversationListResponse, summary="List all conversations")
async def list_conversations():
    """List all active conversation IDs."""
    buffer = get_session_buffer()
    conversation_ids = list(buffer.buffers.keys())
    return ConversationListResponse(conversations=conversation_ids)


@router.delete("/{conversation_id}", summary="Clear conversation history")
async def clear_conversation(conversation_id: str):
    """Clear the session history for a specific conversation."""
    buffer = get_session_buffer()
    if conversation_id in buffer.buffers:
        del buffer.buffers[conversation_id]
        return {"message": f"Conversation '{conversation_id}' cleared"}
    raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found")


@router.delete("", summary="Clear all conversations")
async def clear_all_conversations():
    """Clear all conversation histories."""
    buffer = get_session_buffer()
    count = len(buffer.buffers)
    buffer.buffers.clear()
    return {"message": f"Cleared {count} conversations"}


@router.get("/{conversation_id}/memories", response_model=MemorySearchResponse, summary="Get conversation memories")
async def get_conversation_memories(conversation_id: str, top_k: int = 10):
    """Retrieve long-term memories associated with a conversation."""
    long_term = get_long_term_memory()
    # Search memories filtered by conversation_id
    hits = long_term.store.query(
        query="",  # Empty query to get all
        top_k=top_k,
        where={"conversation_id": conversation_id},
        collection="memories",
    )
    return MemorySearchResponse(results=[hit.model_dump() for hit in hits])


@router.post("/{conversation_id}/summarize", summary="Summarize and store conversation")
async def summarize_conversation(conversation_id: str):
    """Manually trigger summarization of the current conversation to long-term memory."""
    buffer = get_session_buffer()
    long_term = get_long_term_memory()
    
    messages = buffer.get(conversation_id)
    if not messages:
        raise HTTPException(status_code=404, detail=f"No messages found for conversation '{conversation_id}'")
    
    result = await long_term.summarize_and_store(conversation_id, messages)
    if result:
        return {
            "message": "Conversation summarized and stored",
            "memory_id": result.id,
            "summary": result.content,
        }
    raise HTTPException(status_code=500, detail="Failed to summarize conversation")


@router.delete("/{conversation_id}/memories", summary="Clear conversation memories")
async def clear_conversation_memories(conversation_id: str):
    """Clear all long-term memories for a specific conversation."""
    from app.rag.vector_store import get_vector_store
    
    store = get_vector_store()
    try:
        # Get all memories for this conversation
        results = store.memories.get(where={"conversation_id": conversation_id})
        ids = results.get("ids", [])
        if ids:
            store.memories.delete(ids=ids)
            return {"message": f"Cleared {len(ids)} memories for conversation '{conversation_id}'"}
        return {"message": f"No memories found for conversation '{conversation_id}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memories: {str(e)}")
