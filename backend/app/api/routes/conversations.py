from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.memory.session_buffer import get_session_buffer
from app.memory.long_term import get_long_term_memory
from app.models.schemas import ChatMessage
from app.auth.dependencies import get_current_user


router = APIRouter()

# Store conversation metadata (in production, use a database)
conversation_metadata: Dict[str, Dict] = {}


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"


class ConversationInfo(BaseModel):
    id: str
    title: str
    created_at: str
    message_count: int


class ConversationHistory(BaseModel):
    conversation_id: str
    title: str
    messages: List[ChatMessage]
    message_count: int


class ConversationListResponse(BaseModel):
    conversations: List[ConversationInfo]


class MemorySearchResponse(BaseModel):
    results: List[dict]


@router.post("", response_model=ConversationInfo, summary="Create a new conversation")
async def create_conversation(
    data: ConversationCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    conversation_metadata[conversation_id] = {
        "id": conversation_id,
        "title": data.title or "New Conversation",
        "created_at": now,
        "user_id": current_user.get("user_id"),
    }
    
    return ConversationInfo(
        id=conversation_id,
        title=data.title or "New Conversation",
        created_at=now,
        message_count=0,
    )


@router.get("/{conversation_id}", response_model=ConversationHistory, summary="Get conversation history")
async def get_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get the current session history for a conversation."""
    buffer = get_session_buffer()
    messages = buffer.get(conversation_id)
    
    # Get title from metadata or use default
    meta = conversation_metadata.get(conversation_id, {})
    title = meta.get("title", "Conversation")
    
    # Update title based on first user message if still default
    if title == "New Conversation" and messages:
        for msg in messages:
            if msg.role == "user":
                title = msg.content[:50] + ("..." if len(msg.content) > 50 else "")
                if conversation_id in conversation_metadata:
                    conversation_metadata[conversation_id]["title"] = title
                break
    
    return ConversationHistory(
        conversation_id=conversation_id,
        title=title,
        messages=messages,
        message_count=len(messages),
    )


@router.get("", summary="List all conversations")
async def list_conversations(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List all active conversations with metadata."""
    buffer = get_session_buffer()
    conversation_ids = list(buffer.buffers.keys())
    
    result = []
    for conv_id in conversation_ids:
        messages = buffer.get(conv_id)
        meta = conversation_metadata.get(conv_id, {})
        
        # Generate title from first user message if not set
        title = meta.get("title", "Conversation")
        if title in ["New Conversation", "Conversation"] and messages:
            for msg in messages:
                if msg.role == "user":
                    title = msg.content[:50] + ("..." if len(msg.content) > 50 else "")
                    break
        
        result.append({
            "id": conv_id,
            "title": title,
            "created_at": meta.get("created_at", ""),
            "message_count": len(messages),
        })
    
    return result


@router.delete("/{conversation_id}", summary="Clear conversation history")
async def clear_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Clear the session history for a specific conversation."""
    buffer = get_session_buffer()
    if conversation_id in buffer.buffers:
        del buffer.buffers[conversation_id]
        if conversation_id in conversation_metadata:
            del conversation_metadata[conversation_id]
        return {"message": f"Conversation '{conversation_id}' cleared"}
    raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found")


@router.delete("", summary="Clear all conversations")
async def clear_all_conversations(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Clear all conversation histories."""
    buffer = get_session_buffer()
    count = len(buffer.buffers)
    buffer.buffers.clear()
    conversation_metadata.clear()
    return {"message": f"Cleared {count} conversations"}


@router.get("/{conversation_id}/memories", response_model=MemorySearchResponse, summary="Get conversation memories")
async def get_conversation_memories(
    conversation_id: str,
    top_k: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
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
async def summarize_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
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
async def clear_conversation_memories(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
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
