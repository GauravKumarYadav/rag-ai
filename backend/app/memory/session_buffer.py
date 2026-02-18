"""
Session Buffer with Circuit Breaker and Memory-Bounded Fallback.

Production-grade session buffer with:
- Circuit breaker pattern for Redis (no silent failures)
- Memory-bounded fallback during Redis outages
- Automatic recovery with health checks
- LRU eviction for memory management
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Deque, Dict, List, Optional, Tuple, Any

import redis

from app.config import settings
from app.models.schemas import ChatMessage

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation - Redis available
    OPEN = auto()        # Failing - using fallback
    HALF_OPEN = auto()   # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening circuit
    recovery_timeout: float = 30.0      # Seconds before attempting recovery
    half_open_max_calls: int = 3        # Successful calls to close circuit
    health_check_interval: float = 10.0  # Seconds between health checks


class RedisCircuitBreaker:
    """
    Circuit breaker for Redis with automatic recovery.
    
    Prevents cascade failures and provides graceful degradation
    to in-memory storage when Redis is unavailable.
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._redis: Optional[redis.Redis] = None
        self._last_health_check = 0.0
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.state != CircuitState.OPEN:
            return False
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def record_success(self) -> None:
        """Record a successful Redis operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.half_open_max_calls:
                logger.info("Circuit breaker closed - Redis recovered")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            # Decay failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self) -> None:
        """Record a failed Redis operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker opened - Redis recovery failed")
            self.state = CircuitState.OPEN
            self.success_count = 0
        elif self.failure_count >= self.config.failure_threshold:
            if self.state == CircuitState.CLOSED:
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
                self.state = CircuitState.OPEN
    
    def check_health(self, redis_client: Optional[redis.Redis]) -> bool:
        """
        Perform health check on Redis connection.
        Called periodically to detect recovery.
        """
        now = time.time()
        if now - self._last_health_check < self.config.health_check_interval:
            return self.state == CircuitState.CLOSED
        
        self._last_health_check = now
        
        if redis_client is None:
            return False
        
        try:
            redis_client.ping()
            return True
        except Exception:
            return False
    
    def get_state(self) -> CircuitState:
        """Get current circuit state, checking for recovery if needed."""
        if self.state == CircuitState.OPEN and self._should_attempt_recovery():
            logger.info("Attempting Redis recovery - entering half-open state")
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
        return self.state
    
    def is_redis_available(self, redis_client: Optional[redis.Redis]) -> bool:
        """Check if Redis operations should be attempted."""
        state = self.get_state()
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            # Allow limited operations in half-open state
            return True
        else:  # OPEN
            return False


class MemoryBoundedBuffer:
    """
    In-memory buffer with LRU eviction and size bounds.
    
    Provides bounded memory usage during Redis outages with
    automatic eviction of least recently used conversations.
    """
    
    def __init__(
        self,
        max_conversations: int = 1000,
        max_messages_per_conv: int = 100,
        max_tokens: int = 2500,
    ):
        self.max_conversations = max_conversations
        self.max_messages_per_conv = max_messages_per_conv
        self.max_tokens = max_tokens
        self._buffers: Dict[str, Deque[ChatMessage]] = {}
        self._access_times: Dict[str, float] = {}
        self._token_counts: Dict[str, int] = {}
    
    def add(self, conversation_id: str, message: ChatMessage) -> None:
        """Add a message with LRU eviction."""
        # Evict old conversations if at capacity and this is a new conversation
        if conversation_id not in self._buffers and len(self._buffers) >= self.max_conversations:
            self._evict_oldest()
        
        # Initialize buffer if needed
        if conversation_id not in self._buffers:
            self._buffers[conversation_id] = deque(maxlen=self.max_messages_per_conv)
            self._token_counts[conversation_id] = 0
        
        buffer = self._buffers[conversation_id]
        
        # Check token limit before adding
        msg_tokens = self._estimate_tokens(message.content)
        current_tokens = self._token_counts.get(conversation_id, 0)
        
        # Trim if adding would exceed token limit
        while buffer and (current_tokens + msg_tokens) > self.max_tokens:
            removed = buffer.popleft()
            removed_tokens = self._estimate_tokens(removed.content)
            current_tokens -= removed_tokens
        
        # Add the message
        buffer.append(message)
        self._token_counts[conversation_id] = current_tokens + msg_tokens
        self._access_times[conversation_id] = time.time()
    
    def get(self, conversation_id: str) -> List[ChatMessage]:
        """Get messages and update access time."""
        self._access_times[conversation_id] = time.time()
        return list(self._buffers.get(conversation_id, []))
    
    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation from memory."""
        deleted = False
        if conversation_id in self._buffers:
            del self._buffers[conversation_id]
            deleted = True
        self._access_times.pop(conversation_id, None)
        self._token_counts.pop(conversation_id, None)
        return deleted
    
    def clear(self) -> int:
        """Clear all conversations."""
        count = len(self._buffers)
        self._buffers.clear()
        self._access_times.clear()
        self._token_counts.clear()
        return count
    
    def list_conversations(self) -> List[str]:
        """List all conversation IDs in memory."""
        return list(self._buffers.keys())
    
    def _evict_oldest(self) -> None:
        """Evict least recently used conversation."""
        if not self._access_times:
            return
        
        oldest_id = min(self._access_times, key=self._access_times.get)
        logger.info(f"Evicting conversation {oldest_id} due to memory pressure")
        
        del self._buffers[oldest_id]
        del self._access_times[oldest_id]
        del self._token_counts[oldest_id]
    
    @staticmethod
    def _estimate_tokens(content: str) -> int:
        """Fast token estimation."""
        return len(content) // 4  # ~4 chars per token for English


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
        redis_required: bool = False,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.redis_required = redis_required
        
        # Sliding window configuration
        self.sliding_window_turns = sliding_window_turns or settings.session.sliding_window_turns
        self.sliding_window_messages = self.sliding_window_turns * 2
        
        # Initialize circuit breaker and fallback buffer
        self.circuit_breaker = RedisCircuitBreaker()
        self.memory_fallback = MemoryBoundedBuffer(
            max_conversations=1000,
            max_messages_per_conv=max_messages,
            max_tokens=max_tokens,
        )
        
        # Redis connection
        self._redis: Optional[redis.Redis] = None
        self._init_redis()
        
        # Running summaries cache
        self._running_summaries: Dict[str, str] = {}

    def _init_redis(self) -> None:
        """Initialize Redis connection with circuit breaker."""
        try:
            self._redis = redis.from_url(
                settings.redis.url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self._redis.ping()
            logger.info(f"Redis connected for session storage: {settings.redis.url}")
        except Exception as e:
            if self.redis_required:
                logger.error(f"Redis required but unavailable: {e}")
                raise RuntimeError(f"Redis required but unavailable: {e}")
            logger.warning(f"Redis connection failed, using in-memory fallback: {e}")
            self._redis = None
            self.circuit_breaker.record_failure()

    def _redis_key(self, conversation_id: str) -> str:
        return f"{self.REDIS_KEY_PREFIX}{conversation_id}"

    def _metadata_key(self, conversation_id: str) -> str:
        return f"{self.REDIS_METADATA_PREFIX}{conversation_id}"

    def add(self, conversation_id: str, role: str, content: str) -> None:
        """Add a message with circuit breaker pattern."""
        message = ChatMessage(role=role, content=content)
        
        # Always add to memory fallback (bounded LRU)
        self.memory_fallback.add(conversation_id, message)
        
        # Attempt Redis persistence if circuit allows
        if self.circuit_breaker.is_redis_available(self._redis):
            try:
                self._persist_to_redis(conversation_id)
                self.circuit_breaker.record_success()
            except Exception as e:
                logger.warning(f"Redis persist failed, using memory fallback: {e}")
                self.circuit_breaker.record_failure()
        
        # Periodic health check
        self.circuit_breaker.check_health(self._redis)
    
    def _persist_to_redis(self, conversation_id: str) -> None:
        """Persist conversation to Redis."""
        if not self._redis:
            return
        
        messages = self.memory_fallback.get(conversation_id)
        key = self._redis_key(conversation_id)
        
        # Store as JSON list
        messages_data = [{"role": m.role, "content": m.content} for m in messages]
        self._redis.set(key, json.dumps(messages_data))
        
        # Update metadata (title from first user message)
        meta_key = self._metadata_key(conversation_id)
        if not self._redis.exists(meta_key):
            import datetime
            for m in messages:
                if m.role == "user":
                    title = m.content[:50] + "..." if len(m.content) > 50 else m.content
                    self._redis.hset(meta_key, mapping={
                        "title": title,
                        "created_at": datetime.datetime.utcnow().isoformat()
                    })
                    break

    def get(self, conversation_id: str) -> List[ChatMessage]:
        """Get messages from memory or Redis with circuit breaker."""
        # Check memory fallback first (always available)
        messages = self.memory_fallback.get(conversation_id)
        if messages:
            return messages
        
        # Try loading from Redis if circuit allows
        if self.circuit_breaker.is_redis_available(self._redis):
            try:
                key = self._redis_key(conversation_id)
                data = self._redis.get(key)
                if data:
                    messages_data = json.loads(data)
                    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages_data]
                    
                    # Populate memory fallback for future access
                    for msg in messages:
                        self.memory_fallback.add(conversation_id, msg)
                    
                    self.circuit_breaker.record_success()
                    return messages
            except Exception as e:
                logger.warning(f"Failed to load from Redis: {e}")
                self.circuit_breaker.record_failure()
        
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
        for conv_id, buffer in self.memory_fallback._buffers.items():
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
        
        # Remove from memory fallback
        if self.memory_fallback.delete(conversation_id):
            deleted = True
        
        # Remove running summary from memory
        if conversation_id in self._running_summaries:
            del self._running_summaries[conversation_id]
        
        # Remove from Redis if available
        if self.circuit_breaker.is_redis_available(self._redis):
            try:
                key = self._redis_key(conversation_id)
                meta_key = self._metadata_key(conversation_id)
                summary_key = f"{self.REDIS_SUMMARY_PREFIX}{conversation_id}"
                self._redis.delete(key, meta_key, summary_key)
                deleted = True
                self.circuit_breaker.record_success()
            except Exception as e:
                logger.warning(f"Failed to delete from Redis: {e}")
                self.circuit_breaker.record_failure()
        
        return deleted

    def clear_all(self) -> int:
        """Clear all conversations and summaries."""
        count = self.memory_fallback.clear()
        self._running_summaries.clear()
        
        # Clear Redis if available
        if self.circuit_breaker.is_redis_available(self._redis):
            try:
                keys = self._redis.keys(f"{self.REDIS_KEY_PREFIX}*")
                meta_keys = self._redis.keys(f"{self.REDIS_METADATA_PREFIX}*")
                summary_keys = self._redis.keys(f"{self.REDIS_SUMMARY_PREFIX}*")
                all_keys = keys + meta_keys + summary_keys
                if all_keys:
                    self._redis.delete(*all_keys)
                count = max(count, len(keys))
                self.circuit_breaker.record_success()
            except Exception as e:
                logger.warning(f"Failed to clear Redis: {e}")
                self.circuit_breaker.record_failure()
        
        return count

    def _trim(self, buffer: Deque[ChatMessage]) -> None:
        """Trim buffer to size limits (legacy compatibility)."""
        while len(buffer) > self.max_messages:
            buffer.popleft()
        while self._estimated_tokens(buffer) > self.max_tokens and buffer:
            buffer.popleft()

    def _estimated_tokens(self, messages) -> int:
        """Estimate token count for messages."""
        return sum(len(message.content) // 4 for message in messages)


@lru_cache(maxsize=1)
def get_session_buffer() -> SessionBuffer:
    """Get or create the SessionBuffer singleton."""
    return SessionBuffer(
        max_tokens=settings.session_max_tokens,
        max_messages=settings.session_max_messages,
        redis_required=getattr(settings, 'redis_required', False),
    )

