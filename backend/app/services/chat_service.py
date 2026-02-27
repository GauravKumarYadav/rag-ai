"""
Chat Service - Simplified with Multi-Agent Architecture.

This service provides the main chat interface using the LangGraph
multi-agent orchestrator for RAG operations.
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, List, Optional, Tuple

from fastapi import BackgroundTasks

from app.clients.lmstudio import LMStudioClient, get_lmstudio_client
from app.config import settings
from app.memory.session_buffer import SessionBuffer, get_session_buffer
from app.memory.summarizer import get_summarizer
from app.models.client import get_client_store
from app.models.schemas import ChatRequest, RetrievalHit
from app.agents.orchestrator import get_orchestrator

logger = logging.getLogger(__name__)


def generate_message_id() -> str:
    """Generate unique message ID for tracking."""
    import uuid
    return str(uuid.uuid4())[:8]


class ChatService:
    """
    Simplified chat service using multi-agent LangGraph orchestrator.
    
    Features:
    - Multi-agent RAG pipeline
    - Auto-summarization when context exceeds threshold
    - Client-scoped and global document retrieval
    - Production-grade streaming with error boundaries
    """
    
    def __init__(
        self,
        lm_client: LMStudioClient,
        session_buffer: SessionBuffer,
    ) -> None:
        self.lm_client = lm_client
        self.session_buffer = session_buffer
        self.summarizer = get_summarizer()
        self.orchestrator = get_orchestrator()
        self.client_store = get_client_store()

    async def handle_chat(
        self, 
        request: ChatRequest, 
        background_tasks: Optional[BackgroundTasks] = None,
        allowed_clients: Optional[set] = None,
    ) -> Tuple[str | AsyncGenerator[str, None], List[RetrievalHit]]:
        """
        Handle chat request using multi-agent orchestrator.
        
        Args:
            request: The chat request
            background_tasks: Optional background tasks for async operations
            allowed_clients: Set of client IDs the user can access
        """
        conversation_id = request.conversation_id
        client_id = request.client_id or "global"
        
        # Get client name for context
        client_name = ""
        if client_id and client_id != "global":
            client = await self.client_store.get(client_id)
            if client:
                client_name = client.name
            else:
                client_name = client_id  # Fallback to ID if client not found
        else:
            client_name = "Global"
        
        # Get conversation context from memory
        context = self.summarizer.get_context_for_agent(conversation_id)
        
        # Run through multi-agent orchestrator
        response, retrieved = await self.orchestrator.run(
            message=request.message,
            client_id=client_id,
            conversation_id=conversation_id,
            conversation_summary=context.get("conversation_summary", ""),
            recent_messages=context.get("recent_messages", []),
            client_name=client_name,
        )
        
        # Handle streaming
        if request.stream:
            async def stream_chunks() -> AsyncGenerator[str, None]:
                """Yield plain text chunks — transport layer adds framing."""
                message_id = generate_message_id()
                
                try:
                    # Stream content in chunks for memory efficiency
                    chunk_size = 100  # characters per chunk
                    for i in range(0, len(response), chunk_size):
                        yield response[i:i + chunk_size]
                    
                except asyncio.CancelledError:
                    logger.info(f"Stream cancelled by client for message {message_id}")
                    raise
                except Exception as e:
                    logger.error(f"Streaming error for message {message_id}: {e}", exc_info=True)
                    raise
                finally:
                    # Always complete post-turn processing, even if streaming failed
                    try:
                        await self._post_turn(request, response, background_tasks)
                    except Exception as e:
                        logger.error(f"Post-turn processing error: {e}")
            
            return stream_chunks(), retrieved
        
        # Non-streaming: update memory synchronously
        await self._post_turn(request, response, background_tasks)
        
        return response, retrieved

    async def _post_turn(
        self,
        request: ChatRequest,
        assistant_text: str,
        background_tasks: Optional[BackgroundTasks],
    ) -> None:
        """
        Post-turn processing: update memory and check for summarization.
        """
        conversation_id = request.conversation_id
        
        # Add to session buffer
        self.session_buffer.add(conversation_id, "user", request.message)
        self.session_buffer.add(conversation_id, "assistant", assistant_text)
        
        # Check if summarization is needed
        if self.summarizer.should_summarize(conversation_id, self.session_buffer):
            if background_tasks:
                background_tasks.add_task(
                    self.summarizer.summarize,
                    conversation_id,
                    self.session_buffer,
                )
            else:
                await self.summarizer.summarize(conversation_id, self.session_buffer)


# Singleton instance
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get or create the ChatService singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService(
            lm_client=get_lmstudio_client(),
            session_buffer=get_session_buffer(),
        )
    return _chat_service
