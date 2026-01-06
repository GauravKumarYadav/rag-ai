import json
import logging
from collections import defaultdict, deque
from functools import lru_cache
from typing import Deque, Dict, Iterable, List, Optional

import redis

from app.config import settings
from app.models.schemas import ChatMessage

logger = logging.getLogger(__name__)


class SessionBuffer:
    """Session buffer with Redis persistence for conversation history."""
    
    REDIS_KEY_PREFIX = "session:"
    REDIS_METADATA_PREFIX = "session_meta:"
    
    def __init__(self, max_tokens: int, max_messages: int) -> None:
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.buffers: Dict[str, Deque[ChatMessage]] = defaultdict(deque)
        self._redis: Optional[redis.Redis] = None
        self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            self._redis = redis.from_url(settings.redis.url, decode_responses=True)
            self._redis.ping()
            logger.info(f"Redis connected for session storage: {settings.redis.url}")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory storage: {e}")
            self._redis = None

    def _redis_key(self, conversation_id: str) -> str:
        return f"{self.REDIS_KEY_PREFIX}{conversation_id}"

    def _metadata_key(self, conversation_id: str) -> str:
        return f"{self.REDIS_METADATA_PREFIX}{conversation_id}"

    def add(self, conversation_id: str, role: str, content: str) -> None:
        """Add a message to the conversation buffer."""
        message = ChatMessage(role=role, content=content)
        
        # Add to in-memory buffer
        buffer = self.buffers[conversation_id]
        buffer.append(message)
        self._trim(buffer)
        
        # Persist to Redis
        if self._redis:
            try:
                key = self._redis_key(conversation_id)
                # Store as JSON list
                messages_data = [{"role": m.role, "content": m.content} for m in buffer]
                self._redis.set(key, json.dumps(messages_data))
                
                # Update metadata (title from first user message)
                meta_key = self._metadata_key(conversation_id)
                if not self._redis.exists(meta_key):
                    import datetime
                    title = content[:50] + "..." if len(content) > 50 else content
                    if role == "user":
                        self._redis.hset(meta_key, mapping={
                            "title": title,
                            "created_at": datetime.datetime.utcnow().isoformat()
                        })
            except Exception as e:
                logger.error(f"Failed to persist to Redis: {e}")

    def get(self, conversation_id: str) -> List[ChatMessage]:
        """Get messages from a conversation, loading from Redis if not in memory."""
        # Check in-memory first
        if conversation_id in self.buffers:
            return list(self.buffers[conversation_id])
        
        # Try loading from Redis
        if self._redis:
            try:
                key = self._redis_key(conversation_id)
                data = self._redis.get(key)
                if data:
                    messages_data = json.loads(data)
                    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages_data]
                    # Cache in memory
                    self.buffers[conversation_id] = deque(messages)
                    return messages
            except Exception as e:
                logger.error(f"Failed to load from Redis: {e}")
        
        return []

    def list_conversations(self) -> List[Dict]:
        """List all conversations with metadata."""
        conversations = []
        
        # Get from Redis
        if self._redis:
            try:
                # Get all session keys
                keys = self._redis.keys(f"{self.REDIS_KEY_PREFIX}*")
                for key in keys:
                    conv_id = key.replace(self.REDIS_KEY_PREFIX, "")
                    meta_key = self._metadata_key(conv_id)
                    
                    # Get metadata
                    meta = self._redis.hgetall(meta_key) or {}
                    
                    # Get message count
                    data = self._redis.get(key)
                    msg_count = len(json.loads(data)) if data else 0
                    
                    # Get title from first user message if not in metadata
                    title = meta.get("title", "")
                    if not title and data:
                        messages = json.loads(data)
                        for m in messages:
                            if m.get("role") == "user":
                                content = m.get("content", "")
                                title = content[:50] + "..." if len(content) > 50 else content
                                break
                    
                    conversations.append({
                        "id": conv_id,
                        "title": title or "Untitled",
                        "created_at": meta.get("created_at", ""),
                        "message_count": msg_count
                    })
            except Exception as e:
                logger.error(f"Failed to list conversations from Redis: {e}")
        
        # Also include in-memory conversations not in Redis
        for conv_id, buffer in self.buffers.items():
            if not any(c["id"] == conv_id for c in conversations):
                messages = list(buffer)
                title = ""
                for m in messages:
                    if m.role == "user":
                        title = m.content[:50] + "..." if len(m.content) > 50 else m.content
                        break
                conversations.append({
                    "id": conv_id,
                    "title": title or "Untitled",
                    "created_at": "",
                    "message_count": len(messages)
                })
        
        return conversations

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        deleted = False
        
        # Remove from memory
        if conversation_id in self.buffers:
            del self.buffers[conversation_id]
            deleted = True
        
        # Remove from Redis
        if self._redis:
            try:
                key = self._redis_key(conversation_id)
                meta_key = self._metadata_key(conversation_id)
                self._redis.delete(key, meta_key)
                deleted = True
            except Exception as e:
                logger.error(f"Failed to delete from Redis: {e}")
        
        return deleted

    def clear_all(self) -> int:
        """Clear all conversations."""
        count = len(self.buffers)
        self.buffers.clear()
        
        if self._redis:
            try:
                keys = self._redis.keys(f"{self.REDIS_KEY_PREFIX}*")
                meta_keys = self._redis.keys(f"{self.REDIS_METADATA_PREFIX}*")
                all_keys = keys + meta_keys
                if all_keys:
                    self._redis.delete(*all_keys)
                count = max(count, len(keys))
            except Exception as e:
                logger.error(f"Failed to clear Redis: {e}")
        
        return count

    def _trim(self, buffer: Deque[ChatMessage]) -> None:
        while len(buffer) > self.max_messages:
            buffer.popleft()
        while self._estimated_tokens(buffer) > self.max_tokens and buffer:
            buffer.popleft()

    def _estimated_tokens(self, messages: Iterable[ChatMessage]) -> int:
        return sum(len(message.content.split()) for message in messages)


@lru_cache(maxsize=1)
def get_session_buffer() -> SessionBuffer:
    return SessionBuffer(
        max_tokens=settings.session_max_tokens,
        max_messages=settings.session_max_messages,
    )

