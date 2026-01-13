"""
Query Processing Pipeline for Small Model RAG Optimization.

This module handles:
1. Intent classification (chitchat vs question vs follow-up vs action)
2. Reference resolution ("that doc", "as we discussed")
3. Query rewriting for optimal retrieval

By routing and rewriting queries before retrieval, we avoid
unnecessary RAG calls and improve retrieval quality.
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Set

from app.memory.conversation_state import ConversationState

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Classification of user intent."""
    CHITCHAT = "chitchat"           # Greetings, small talk - no retrieval
    QUESTION = "question"           # Direct question - needs retrieval
    FOLLOW_UP = "follow_up"         # Follow-up on previous context - resolve refs first
    ACTION_REQUEST = "action"       # Request to do something - may need retrieval
    CLARIFICATION = "clarification" # Asking for clarification - use state


@dataclass
class QueryResult:
    """Result of query processing pipeline."""
    intent: Intent
    needs_retrieval: bool
    original_query: str
    search_query: str  # Rewritten query for retrieval
    resolved_references: List[str]  # References that were resolved
    confidence: float  # Confidence in classification


# Patterns for intent classification
CHITCHAT_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings|yo)[\s!?.]*$",
    r"^(how are you|how's it going|what's up|sup)[\s!?.]*$",
    r"^(good morning|good afternoon|good evening|good night)[\s!?.]*$",
    r"^(thanks|thank you|thx|ty|appreciate it)[\s!?.]*$",
    r"^(bye|goodbye|see you|later|cya|farewell)[\s!?.]*$",
    r"^(yes|no|ok|okay|sure|alright|fine|great|cool|got it)[\s!?.]*$",
    r"^(help|what can you do|who are you)[\s!?.]*$",
    r"^(nice|awesome|perfect|excellent)[\s!?.]*$",
]

# Patterns indicating follow-up/reference to previous context
REFERENCE_PATTERNS = [
    (r"\b(that|those|the)\s+(doc|document|file|pdf|report)s?\b", "document_ref"),
    (r"\b(as we|like we|that we)\s+(discussed|mentioned|talked about|said)\b", "discussion_ref"),
    (r"\b(those|the)\s+(numbers?|figures?|stats?|statistics?|data)\b", "data_ref"),
    (r"\b(it|this|that)\s+(says?|shows?|mentions?|indicates?)\b", "content_ref"),
    (r"\b(the same|same)\s+(one|thing|document|file)\b", "same_ref"),
    (r"\b(previous|earlier|before|last)\s+(question|query|request|message)\b", "previous_ref"),
    (r"\bwhat about\b", "continuation_ref"),
    (r"\band\s+(also|what about|how about)\b", "continuation_ref"),
    (r"\bcan you (also|elaborate|explain more)\b", "elaboration_ref"),
]

# Keywords suggesting document/retrieval is needed
RETRIEVAL_KEYWORDS = {
    "document", "file", "pdf", "report", "record", "show", "find", "search",
    "look up", "information", "details", "data", "analyze", "summary",
    "summarize", "explain", "list", "get", "retrieve", "what is", "what are",
    "how does", "how do", "why does", "why do", "when did", "where is",
    "tell me about", "describe", "define", "meaning of",
}

# Action request patterns
ACTION_PATTERNS = [
    r"^(please\s+)?(can you|could you|would you|will you)",
    r"^(please\s+)?(create|make|generate|write|build|send|update|delete|remove)",
    r"^(please\s+)?(format|convert|translate|rephrase|rewrite)",
    r"^(please\s+)?(save|store|remember|note)",
]


class QueryProcessor:
    """
    Processes user queries to determine intent, resolve references,
    and rewrite for optimal retrieval.
    """
    
    def __init__(self) -> None:
        self._chitchat_patterns = [re.compile(p, re.IGNORECASE) for p in CHITCHAT_PATTERNS]
        self._reference_patterns = [(re.compile(p, re.IGNORECASE), name) for p, name in REFERENCE_PATTERNS]
        self._action_patterns = [re.compile(p, re.IGNORECASE) for p in ACTION_PATTERNS]
    
    async def process(
        self, 
        message: str, 
        state: ConversationState,
        lm_client: Optional[object] = None,
    ) -> QueryResult:
        """
        Process a user message through the query pipeline.
        
        Args:
            message: The user's message
            state: Current conversation state
            lm_client: Optional LLM client for complex rewrites
            
        Returns:
            QueryResult with intent, retrieval flag, and optimized search query
        """
        original = message.strip()
        
        # Step 1: Classify intent
        intent, confidence = self._classify_intent(original, state)
        
        # Step 2: Check if retrieval is needed based on intent
        needs_retrieval = self._needs_retrieval(intent, original)
        
        # Step 3: Resolve references if this is a follow-up
        resolved_refs = []
        resolved_message = original
        if intent == Intent.FOLLOW_UP:
            resolved_message, resolved_refs = self._resolve_references(original, state)
        
        # Step 4: Rewrite query for retrieval
        search_query = original
        if needs_retrieval:
            search_query = self._rewrite_for_retrieval(resolved_message, state)
        
        logger.debug(f"Query processed: intent={intent.value}, retrieval={needs_retrieval}, "
                    f"original='{original[:50]}...', search='{search_query[:50]}...'")
        
        return QueryResult(
            intent=intent,
            needs_retrieval=needs_retrieval,
            original_query=original,
            search_query=search_query,
            resolved_references=resolved_refs,
            confidence=confidence,
        )
    
    def _classify_intent(self, message: str, state: ConversationState) -> tuple[Intent, float]:
        """
        Classify the intent of a message.
        
        Returns (Intent, confidence) tuple.
        """
        msg_lower = message.lower().strip()
        
        # Check for chitchat patterns (high confidence)
        for pattern in self._chitchat_patterns:
            if pattern.match(msg_lower):
                return Intent.CHITCHAT, 0.95
        
        # Check for reference patterns (follow-up)
        ref_matches = []
        for pattern, ref_type in self._reference_patterns:
            if pattern.search(msg_lower):
                ref_matches.append(ref_type)
        
        if ref_matches:
            # Has references to previous context
            return Intent.FOLLOW_UP, 0.85
        
        # Check for action patterns
        for pattern in self._action_patterns:
            if pattern.match(msg_lower):
                # Check if it needs retrieval context
                if self._has_retrieval_keywords(msg_lower):
                    return Intent.ACTION_REQUEST, 0.8
                # Pure action without document context
                return Intent.ACTION_REQUEST, 0.75
        
        # Check for question indicators
        if "?" in message or self._has_retrieval_keywords(msg_lower):
            return Intent.QUESTION, 0.8
        
        # Short messages without context keywords likely don't need retrieval
        if len(msg_lower) < 20 and not self._has_retrieval_keywords(msg_lower):
            return Intent.CHITCHAT, 0.6
        
        # Default to question (safer to retrieve than miss)
        return Intent.QUESTION, 0.5
    
    def _has_retrieval_keywords(self, message: str) -> bool:
        """Check if message contains keywords suggesting retrieval is needed."""
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in RETRIEVAL_KEYWORDS)
    
    def _needs_retrieval(self, intent: Intent, message: str) -> bool:
        """Determine if retrieval is needed based on intent."""
        if intent == Intent.CHITCHAT:
            return False
        
        if intent == Intent.CLARIFICATION:
            # Usually can answer from state
            return False
        
        if intent in (Intent.QUESTION, Intent.FOLLOW_UP):
            return True
        
        if intent == Intent.ACTION_REQUEST:
            # Check if the action needs document context
            return self._has_retrieval_keywords(message.lower())
        
        return True  # Default to retrieval
    
    def _resolve_references(
        self, 
        message: str, 
        state: ConversationState
    ) -> tuple[str, List[str]]:
        """
        Resolve references in the message using conversation state.
        
        Examples:
            "What about that document?" -> "What about [doc_name from state]?"
            "Tell me more about those numbers" -> "Tell me more about [data from state]"
        
        Returns:
            (resolved_message, list_of_resolved_references)
        """
        resolved = message
        resolved_refs = []
        
        # Try to resolve document references
        if state.entities:
            # Find document-related entities
            doc_entities = {k: v for k, v in state.entities.items() 
                          if any(t in v.lower() for t in ["document", "file", "pdf", "report"])}
            
            if doc_entities:
                # Replace generic document references with specific ones
                doc_name = list(doc_entities.keys())[0]  # Most recent
                for pattern in [r"\b(that|the)\s+doc(ument)?\b", r"\b(that|the)\s+file\b"]:
                    if re.search(pattern, resolved, re.IGNORECASE):
                        resolved = re.sub(pattern, doc_name, resolved, flags=re.IGNORECASE)
                        resolved_refs.append(f"document -> {doc_name}")
        
        # Resolve "as we discussed" type references using running summary
        if state.running_summary:
            discussion_pattern = r"\b(as we|like we|that we)\s+(discussed|mentioned|talked about)\b"
            if re.search(discussion_pattern, resolved, re.IGNORECASE):
                # Append context from running summary
                context_hint = f" (context: {state.running_summary[:100]}...)"
                resolved_refs.append("discussion_ref -> running_summary")
        
        # Resolve client context references
        if state.client_context:
            client_pattern = r"\b(that|the)\s+(client|customer|account)\b"
            if re.search(client_pattern, resolved, re.IGNORECASE):
                resolved = re.sub(client_pattern, state.client_context, resolved, flags=re.IGNORECASE)
                resolved_refs.append(f"client -> {state.client_context}")
        
        # Resolve "those numbers/data" references from entities
        if state.entities:
            data_entities = {k: v for k, v in state.entities.items()
                           if any(t in v.lower() for t in ["number", "data", "figure", "stat"])}
            if data_entities:
                for pattern in [r"\b(those|the)\s+(numbers?|figures?|data)\b"]:
                    if re.search(pattern, resolved, re.IGNORECASE):
                        data_name = list(data_entities.keys())[0]
                        resolved = re.sub(pattern, data_name, resolved, flags=re.IGNORECASE)
                        resolved_refs.append(f"data -> {data_name}")
        
        return resolved, resolved_refs
    
    def _rewrite_for_retrieval(self, message: str, state: ConversationState) -> str:
        """
        Rewrite a message into an optimal search query.
        
        This converts conversational queries into standalone search queries
        that work better with vector similarity search.
        
        Examples:
            "Can you tell me about it?" -> "information about [entity]"
            "What's the policy for that?" -> "[entity] policy"
        """
        query = message
        
        # Remove conversational prefixes
        prefixes_to_remove = [
            r"^(please\s+)?(can you|could you|would you)\s+",
            r"^(please\s+)?(tell me|show me|find|search for)\s+",
            r"^(i want to know|i need to know|i'm looking for)\s+",
            r"^(what is|what are|what's)\s+",
        ]
        
        for prefix in prefixes_to_remove:
            query = re.sub(prefix, "", query, flags=re.IGNORECASE).strip()
        
        # Remove trailing question marks and common endings
        query = re.sub(r"\?+$", "", query).strip()
        query = re.sub(r"\s+(please|thanks|thank you)$", "", query, flags=re.IGNORECASE).strip()
        
        # Add context from state for better retrieval
        context_additions = []
        
        # Add current task context if relevant
        if state.current_task and len(query) < 50:
            # Check if query is too generic
            if len(query.split()) < 4:
                context_additions.append(state.current_task)
        
        # Add client context for scoping
        if state.client_context:
            context_additions.append(state.client_context)
        
        # Add goal context if query seems incomplete
        if state.user_goal and len(query.split()) < 3:
            context_additions.append(state.user_goal)
        
        # Combine query with context
        if context_additions:
            # Prepend context for better embedding match
            context_str = " ".join(context_additions[:2])  # Limit context
            query = f"{context_str} {query}"
        
        # Clean up and normalize
        query = re.sub(r"\s+", " ", query).strip()
        
        # Ensure minimum query length
        if len(query) < 3:
            query = message  # Fall back to original
        
        return query
    
    def classify_simple(self, message: str) -> tuple[bool, str]:
        """
        Simple classification for backward compatibility.
        
        Returns (needs_retrieval, intent_name) tuple.
        """
        msg_lower = message.lower().strip()
        
        # Check chitchat patterns
        for pattern in self._chitchat_patterns:
            if pattern.match(msg_lower):
                return False, "chitchat"
        
        # Check for retrieval keywords
        if self._has_retrieval_keywords(msg_lower):
            return True, "question"
        
        # Short messages without keywords
        if len(msg_lower) < 20 and "?" not in msg_lower:
            return False, "chitchat"
        
        # Default: assume retrieval needed
        return True, "question"


@lru_cache(maxsize=1)
def get_query_processor() -> QueryProcessor:
    """Get singleton query processor instance."""
    return QueryProcessor()
