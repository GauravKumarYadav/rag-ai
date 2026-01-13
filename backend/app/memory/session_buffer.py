"""
Session Buffer with Sliding Window Support for Small Model RAG.

Enhanced session buffer that supports:
- Sliding window (last N turns) for context-limited models
- Running summary generation for older history
- Integration with conversation state management
"""

import json
import logging
from collections import defaultdict, deque
from functools import lru_cache
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import redis

from app.config import settings
from app.models.schemas import ChatMessage

logger = logging.getLogger(__name__)


class SessionBuffer:
    """
    Session buffer with Redis persistence and sliding window support.
    
    For small model optimization:
    - get_recent(): Returns only last N turns (sliding window)
    - get_older(): Returns messages outside the window (for summarization)
    - Full history is still available via get() for backward compatibility
    """
    
    REDIS_KEY_PREFIX = "session:"
    REDIS_METADATA_PREFIX = "session_meta:"
    REDIS_SUMMARY_PREFIX = "session_summary:"
    
    def __init__(
        self, 
        max_tokens: int, 
        max_messages: int,
        sliding_window_turns: Optional[int] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        # Sliding window: number of turns (user+assistant pairs) to keep in context
        self.sliding_window_turns = sliding_window_turns or settings.session.sliding_window_turns
        # Convert turns to messages (2 messages per turn)
        self.sliding_window_messages = self.sliding_window_turns * 2
        self.buffers: Dict[str, Deque[ChatMessage]] = defaultdict(deque)
        self._running_summaries: Dict[str, str] = {}  # Cache for running summaries
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
        """Get all messages from a conversation, loading from Redis if not in memory."""
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
    
    def get_recent(self, conversation_id: str, n: Optional[int] = None) -> List[ChatMessage]:
        """
        Get only the most recent messages (sliding window).
        
        For small model optimization - returns only last N messages
        to avoid context overflow.
        
        Args:
            conversation_id: The conversation ID
            n: Number of messages to return. If None, uses sliding_window_messages.
            
        Returns:
            Last N messages from the conversation
        """
        n = n or self.sliding_window_messages
        all_messages = self.get(conversation_id)
        return all_messages[-n:] if len(all_messages) > n else all_messages
    
    def get_older(self, conversation_id: str, n: Optional[int] = None) -> List[ChatMessage]:
        """
        Get messages outside the sliding window (older history).
        
        These are candidates for summarization into running summary.
        
        Args:
            conversation_id: The conversation ID
            n: Size of sliding window. If None, uses sliding_window_messages.
            
        Returns:
            Messages older than the sliding window
        """
        n = n or self.sliding_window_messages
        all_messages = self.get(conversation_id)
        if len(all_messages) > n:
            return all_messages[:-n]
        return []
    
    def get_with_summary(
        self, 
        conversation_id: str,
        n: Optional[int] = None,
    ) -> Tuple[List[ChatMessage], Optional[str]]:
        """
        Get recent messages and running summary for older history.
        
        This is the recommended method for small model RAG - provides
        recent context plus a summary of older history.
        
        Args:
            conversation_id: The conversation ID
            n: Size of sliding window
            
        Returns:
            Tuple of (recent_messages, running_summary)
        """
        recent = self.get_recent(conversation_id, n)
        summary = self.get_running_summary(conversation_id)
        return recent, summary
    
    def get_running_summary(self, conversation_id: str) -> Optional[str]:
        """
        Get the running summary for older messages.
        
        Returns cached summary if available, otherwise returns None.
        Summaries should be generated externally (e.g., by LongTermMemory).
        """
        # Check memory cache first
        if conversation_id in self._running_summaries:
            return self._running_summaries[conversation_id]
        
        # Try loading from Redis
        if self._redis:
            try:
                key = f"{self.REDIS_SUMMARY_PREFIX}{conversation_id}"
                summary = self._redis.get(key)
                if summary:
                    self._running_summaries[conversation_id] = summary
                    return summary
            except Exception as e:
                logger.error(f"Failed to load summary from Redis: {e}")
        
        return None
    
    def set_running_summary(self, conversation_id: str, summary: str) -> None:
        """
        Set the running summary for older messages.
        
        Called after summarizing older history.
        """
        self._running_summaries[conversation_id] = summary
        
        if self._redis:
            try:
                key = f"{self.REDIS_SUMMARY_PREFIX}{conversation_id}"
                # TTL same as session (24 hours default)
                self._redis.setex(key, settings.redis.session_ttl, summary)
            except Exception as e:
                logger.error(f"Failed to save summary to Redis: {e}")
    
    def needs_summarization(self, conversation_id: str) -> bool:
        """
        Check if older messages need summarization.
        
        Returns True if there are messages outside the sliding window
        that haven't been summarized yet.
        """
        older = self.get_older(conversation_id)
        if not older:
            return False
        
        # Check if we have a summary already
        existing_summary = self.get_running_summary(conversation_id)
        if existing_summary:
            # Summary exists - only need to update if significantly more messages
            return len(older) >= self.sliding_window_messages
        
        # No summary yet - need to create one
        return len(older) >= 2  # At least one turn

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
        """Delete a conversation and its associated data."""
        deleted = False
        
        # Remove from memory
        if conversation_id in self.buffers:
            del self.buffers[conversation_id]
            deleted = True
        
        # Remove running summary from memory
        if conversation_id in self._running_summaries:
            del self._running_summaries[conversation_id]
        
        # Remove from Redis
        if self._redis:
            try:
                key = self._redis_key(conversation_id)
                meta_key = self._metadata_key(conversation_id)
                summary_key = f"{self.REDIS_SUMMARY_PREFIX}{conversation_id}"
                self._redis.delete(key, meta_key, summary_key)
                deleted = True
            except Exception as e:
                logger.error(f"Failed to delete from Redis: {e}")
        
        return deleted

    def clear_all(self) -> int:
        """Clear all conversations and summaries."""
        count = len(self.buffers)
        self.buffers.clear()
        self._running_summaries.clear()
        
        if self._redis:
            try:
                keys = self._redis.keys(f"{self.REDIS_KEY_PREFIX}*")
                meta_keys = self._redis.keys(f"{self.REDIS_METADATA_PREFIX}*")
                summary_keys = self._redis.keys(f"{self.REDIS_SUMMARY_PREFIX}*")
                all_keys = keys + meta_keys + summary_keys
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

