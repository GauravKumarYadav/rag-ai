"""
Structured Conversation State Management.

Instead of passing raw chat history to small models (which confuses them),
we maintain structured state fields that get rendered as a compact "state block".

This gives the model stable memory without long token sequences.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from functools import lru_cache

import redis

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """
    Structured conversation state for small model optimization.
    
    Fields:
        user_goal: The user's primary objective in this conversation
        current_task: What we're currently working on
        entities: Key entities mentioned (people, systems, products, etc.)
        constraints: User-specified constraints (time, budget, format, etc.)
        decisions_made: Important decisions or choices made
        open_questions: Unresolved questions to follow up on
        running_summary: Compact summary of older conversation history
        client_context: Active client/project context if any
    """
    user_goal: Optional[str] = None
    current_task: Optional[str] = None
    entities: Dict[str, str] = field(default_factory=dict)  # name -> type/description
    constraints: List[str] = field(default_factory=list)
    decisions_made: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    running_summary: Optional[str] = None
    client_context: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """Create from dictionary."""
        return cls(
            user_goal=data.get("user_goal"),
            current_task=data.get("current_task"),
            entities=data.get("entities", {}),
            constraints=data.get("constraints", []),
            decisions_made=data.get("decisions_made", []),
            open_questions=data.get("open_questions", []),
            running_summary=data.get("running_summary"),
            client_context=data.get("client_context"),
        )
    
    def is_empty(self) -> bool:
        """Check if state has any meaningful content."""
        return (
            not self.user_goal
            and not self.current_task
            and not self.entities
            and not self.constraints
            and not self.decisions_made
            and not self.open_questions
            and not self.running_summary
            and not self.client_context
        )


class StateBlockBuilder:
    """
    Builds a compact state block (5-15 lines) from ConversationState.
    
    The state block replaces raw chat history in the prompt,
    giving the model structured context without token bloat.
    """
    
    @staticmethod
    def build(state: ConversationState) -> str:
        """
        Build a compact state block string.
        
        Format:
        ```
        [Conversation State]
        Goal: <user_goal>
        Task: <current_task>
        Entities: entity1 (type), entity2 (type)
        Constraints: constraint1, constraint2
        Decisions: decision1; decision2
        Open Questions: question1
        Context: <running_summary>
        Client: <client_context>
        ```
        """
        if state.is_empty():
            return ""
        
        lines = ["[Conversation State]"]
        
        if state.user_goal:
            lines.append(f"Goal: {state.user_goal}")
        
        if state.current_task:
            lines.append(f"Task: {state.current_task}")
        
        if state.entities:
            entity_strs = [f"{name} ({desc})" for name, desc in state.entities.items()]
            # Limit to prevent token bloat
            if len(entity_strs) > 5:
                entity_strs = entity_strs[:5] + [f"... +{len(entity_strs) - 5} more"]
            lines.append(f"Entities: {', '.join(entity_strs)}")
        
        if state.constraints:
            constraints = state.constraints[:5]  # Limit
            lines.append(f"Constraints: {', '.join(constraints)}")
        
        if state.decisions_made:
            decisions = state.decisions_made[-3:]  # Keep most recent
            lines.append(f"Decisions: {'; '.join(decisions)}")
        
        if state.open_questions:
            questions = state.open_questions[:2]  # Limit
            lines.append(f"Open Questions: {'; '.join(questions)}")
        
        if state.running_summary:
            # Truncate if too long
            summary = state.running_summary
            if len(summary) > 200:
                summary = summary[:200] + "..."
            lines.append(f"Context: {summary}")
        
        if state.client_context:
            lines.append(f"Client: {state.client_context}")
        
        return "\n".join(lines)


class ConversationStateManager:
    """
    Manages conversation state persistence and updates.
    
    State is stored in Redis for persistence across requests.
    Falls back to in-memory storage if Redis unavailable.
    """
    
    REDIS_KEY_PREFIX = "conv_state:"
    STATE_TTL = 86400 * 7  # 7 days
    
    def __init__(self) -> None:
        self._redis: Optional[redis.Redis] = None
        self._memory_cache: Dict[str, ConversationState] = {}
        self._init_redis()
        self.block_builder = StateBlockBuilder()
    
    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            self._redis = redis.from_url(settings.redis.url, decode_responses=True)
            self._redis.ping()
            logger.info("Redis connected for conversation state storage")
        except Exception as e:
            logger.warning(f"Redis unavailable for state storage, using memory: {e}")
            self._redis = None
    
    def _redis_key(self, conversation_id: str) -> str:
        """Generate Redis key for conversation state."""
        return f"{self.REDIS_KEY_PREFIX}{conversation_id}"
    
    async def get_state(self, conversation_id: str) -> ConversationState:
        """
        Get conversation state, loading from Redis if not in memory.
        """
        # Check memory cache first
        if conversation_id in self._memory_cache:
            return self._memory_cache[conversation_id]
        
        # Try loading from Redis
        if self._redis:
            try:
                key = self._redis_key(conversation_id)
                data = self._redis.get(key)
                if data:
                    state = ConversationState.from_dict(json.loads(data))
                    self._memory_cache[conversation_id] = state
                    return state
            except Exception as e:
                logger.error(f"Failed to load state from Redis: {e}")
        
        # Return empty state
        state = ConversationState()
        self._memory_cache[conversation_id] = state
        return state
    
    async def save_state(self, conversation_id: str, state: ConversationState) -> None:
        """
        Save conversation state to Redis.
        """
        self._memory_cache[conversation_id] = state
        
        if self._redis:
            try:
                key = self._redis_key(conversation_id)
                self._redis.setex(key, self.STATE_TTL, json.dumps(state.to_dict()))
            except Exception as e:
                logger.error(f"Failed to save state to Redis: {e}")
    
    async def update_state(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        extracted_entities: Optional[Dict[str, str]] = None,
        extracted_decisions: Optional[List[str]] = None,
        new_constraints: Optional[List[str]] = None,
    ) -> ConversationState:
        """
        Update state after a conversation turn.
        
        This is called after each exchange to maintain state.
        Entity/decision extraction can be done by the caller (e.g., via LLM).
        """
        state = await self.get_state(conversation_id)
        
        # Update entities if provided
        if extracted_entities:
            state.entities.update(extracted_entities)
            # Keep entities bounded
            if len(state.entities) > 20:
                # Keep most recently added
                items = list(state.entities.items())
                state.entities = dict(items[-20:])
        
        # Add decisions if provided
        if extracted_decisions:
            state.decisions_made.extend(extracted_decisions)
            # Keep bounded
            state.decisions_made = state.decisions_made[-10:]
        
        # Add constraints if provided
        if new_constraints:
            for c in new_constraints:
                if c not in state.constraints:
                    state.constraints.append(c)
            state.constraints = state.constraints[-10:]
        
        await self.save_state(conversation_id, state)
        return state
    
    async def set_goal(self, conversation_id: str, goal: str) -> None:
        """Set or update the user's goal."""
        state = await self.get_state(conversation_id)
        state.user_goal = goal
        await self.save_state(conversation_id, state)
    
    async def set_task(self, conversation_id: str, task: str) -> None:
        """Set or update the current task."""
        state = await self.get_state(conversation_id)
        state.current_task = task
        await self.save_state(conversation_id, state)
    
    async def add_entity(self, conversation_id: str, name: str, description: str) -> None:
        """Add an entity to the state."""
        state = await self.get_state(conversation_id)
        state.entities[name] = description
        await self.save_state(conversation_id, state)
    
    async def add_decision(self, conversation_id: str, decision: str) -> None:
        """Record a decision made."""
        state = await self.get_state(conversation_id)
        state.decisions_made.append(decision)
        state.decisions_made = state.decisions_made[-10:]  # Keep bounded
        await self.save_state(conversation_id, state)
    
    async def add_open_question(self, conversation_id: str, question: str) -> None:
        """Add an open question."""
        state = await self.get_state(conversation_id)
        state.open_questions.append(question)
        state.open_questions = state.open_questions[-5:]  # Keep bounded
        await self.save_state(conversation_id, state)
    
    async def resolve_question(self, conversation_id: str, question: str) -> None:
        """Remove a resolved question."""
        state = await self.get_state(conversation_id)
        state.open_questions = [q for q in state.open_questions if q != question]
        await self.save_state(conversation_id, state)
    
    async def set_running_summary(self, conversation_id: str, summary: str) -> None:
        """Update the running summary of older history."""
        state = await self.get_state(conversation_id)
        state.running_summary = summary
        await self.save_state(conversation_id, state)
    
    async def set_client_context(self, conversation_id: str, client_context: str) -> None:
        """Set the active client context."""
        state = await self.get_state(conversation_id)
        state.client_context = client_context
        await self.save_state(conversation_id, state)
    
    def build_state_block(self, state: ConversationState) -> str:
        """Build a compact state block string for prompt injection."""
        return self.block_builder.build(state)
    
    async def clear_state(self, conversation_id: str) -> None:
        """Clear state for a conversation."""
        if conversation_id in self._memory_cache:
            del self._memory_cache[conversation_id]
        
        if self._redis:
            try:
                key = self._redis_key(conversation_id)
                self._redis.delete(key)
            except Exception as e:
                logger.error(f"Failed to delete state from Redis: {e}")


@lru_cache(maxsize=1)
def get_state_manager() -> ConversationStateManager:
    """Get singleton state manager instance."""
    return ConversationStateManager()
