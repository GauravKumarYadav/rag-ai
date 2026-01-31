"""
Conversation Summarizer for Auto-Summarization.

This module provides automatic conversation summarization when
the context exceeds the configured threshold.

Features:
- Automatic trigger when context tokens exceed threshold
- Incremental summarization (preserves key facts)
- Redis-persisted summaries
- LLM-based summarization using LM Studio
"""

import logging
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from app.config import settings
from app.memory.session_buffer import SessionBuffer, get_session_buffer
from app.core.cost_tracker import get_cost_tracker

logger = logging.getLogger(__name__)


SUMMARIZATION_PROMPT = """You are a conversation summarizer. Create a concise summary that preserves:

1. Key facts and information discussed
2. Important decisions made
3. Current topic/context of discussion
4. Any specific details the user mentioned (names, numbers, dates)
5. The user's apparent goal or need

Keep the summary factual and concise (under {target_tokens} words).
Do NOT include opinions or interpretations.

{existing_summary_section}

Messages to summarize:
{messages}

Create a summary that would help continue this conversation:"""


class ConversationSummarizer:
    """
    LLM-based conversation summarizer with automatic triggering.
    """
    
    def __init__(self) -> None:
        self.cost_tracker = get_cost_tracker()
        self.llm = ChatOpenAI(
            base_url=settings.llm.lmstudio.base_url,
            model=settings.llm.lmstudio.model,
            api_key="lmstudio",
            temperature=0.3,  # Low temperature for factual summaries
            max_tokens=settings.memory.summary_target_tokens,
            timeout=settings.llm.timeout,
            callbacks=[self.cost_tracker],
        )
        self.max_context_tokens = settings.memory.max_context_tokens
        self.summary_target_tokens = settings.memory.summary_target_tokens
        self.sliding_window_size = settings.memory.sliding_window_size
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses a simple heuristic of ~4 characters per token.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def should_summarize(
        self,
        conversation_id: str,
        buffer: Optional[SessionBuffer] = None,
    ) -> bool:
        """
        Check if conversation needs summarization.
        
        Triggers when:
        1. Total context (summary + recent messages) exceeds max_context_tokens
        2. There are messages outside the sliding window
        
        Args:
            conversation_id: Conversation ID
            buffer: Optional session buffer (uses singleton if not provided)
            
        Returns:
            True if summarization is needed
        """
        if buffer is None:
            buffer = get_session_buffer()
        
        # Get current context
        recent, existing_summary = buffer.get_with_summary(conversation_id)
        
        # Calculate total context tokens
        recent_text = " ".join([m.content for m in recent])
        total_tokens = self.estimate_tokens(recent_text)
        
        if existing_summary:
            total_tokens += self.estimate_tokens(existing_summary)
        
        # Check if we exceed threshold
        if total_tokens > self.max_context_tokens:
            return True
        
        # Check if there are unsummarized older messages
        return buffer.needs_summarization(conversation_id)
    
    @traceable(name="summarizer.summarize")
    async def summarize(
        self,
        conversation_id: str,
        buffer: Optional[SessionBuffer] = None,
    ) -> str:
        """
        Summarize older messages and update the running summary.
        
        Args:
            conversation_id: Conversation ID
            buffer: Optional session buffer
            
        Returns:
            Updated summary
        """
        if buffer is None:
            buffer = get_session_buffer()
        
        # Get messages to summarize
        older_messages = buffer.get_older(conversation_id)
        if not older_messages:
            return buffer.get_running_summary(conversation_id) or ""
        
        # Get existing summary
        existing_summary = buffer.get_running_summary(conversation_id)
        
        # Format messages
        messages_text = self._format_messages(older_messages)
        
        # Build prompt
        existing_section = ""
        if existing_summary:
            existing_section = f"Existing summary to incorporate:\n{existing_summary}\n\n"
        
        prompt = SUMMARIZATION_PROMPT.format(
            target_tokens=self.summary_target_tokens // 4,  # Convert to approximate words
            existing_summary_section=existing_section,
            messages=messages_text,
        )
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a concise conversation summarizer."),
                HumanMessage(content=prompt),
            ])
            
            new_summary = response.content or ""
            
            # Store the summary
            if new_summary:
                buffer.set_running_summary(conversation_id, new_summary)
                logger.info(f"Updated summary for conversation {conversation_id}")
            
            return new_summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return existing_summary or ""
    
    async def summarize_if_needed(
        self,
        conversation_id: str,
        buffer: Optional[SessionBuffer] = None,
    ) -> Optional[str]:
        """
        Check if summarization is needed and perform it if so.
        
        Convenience method that combines should_summarize and summarize.
        
        Args:
            conversation_id: Conversation ID
            buffer: Optional session buffer
            
        Returns:
            New summary if created, None otherwise
        """
        if buffer is None:
            buffer = get_session_buffer()
        
        if self.should_summarize(conversation_id, buffer):
            return await self.summarize(conversation_id, buffer)
        
        return None
    
    def _format_messages(self, messages: List) -> str:
        """
        Format messages for the summarization prompt.
        
        Args:
            messages: List of ChatMessage objects
            
        Returns:
            Formatted string
        """
        lines = []
        for msg in messages:
            role = msg.role.capitalize()
            # Truncate very long messages
            content = msg.content
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def get_context_for_agent(
        self,
        conversation_id: str,
        buffer: Optional[SessionBuffer] = None,
    ) -> Dict[str, any]:
        """
        Get conversation context formatted for agent use.
        
        Args:
            conversation_id: Conversation ID
            buffer: Optional session buffer
            
        Returns:
            Dict with 'summary' and 'recent_messages' keys
        """
        if buffer is None:
            buffer = get_session_buffer()
        
        recent, summary = buffer.get_with_summary(conversation_id)
        
        return {
            "conversation_summary": summary or "",
            "recent_messages": [
                {"role": m.role, "content": m.content}
                for m in recent
            ],
        }


# Singleton instance
_summarizer: Optional[ConversationSummarizer] = None


def get_summarizer() -> ConversationSummarizer:
    """Get or create the ConversationSummarizer singleton."""
    global _summarizer
    if _summarizer is None:
        _summarizer = ConversationSummarizer()
    return _summarizer
