"""
Shared state for multi-agent LangGraph system.

This module defines the AgentState TypedDict that is shared across
all agents in the system. The state is passed through the graph
and updated by each agent.
"""

from typing import Any, Dict, List, Optional, TypedDict

from app.models.schemas import RetrievalHit


class AgentState(TypedDict, total=False):
    """
    Shared state for multi-agent system.
    
    All agents read from and write to this shared state.
    Uses total=False to make all fields optional for partial updates.
    """
    
    # ==========================================================================
    # Input Fields
    # ==========================================================================
    message: str  # User's input message
    client_id: str  # Client ID for collection scoping
    client_name: str  # Client display name for context
    conversation_id: str  # Conversation ID for memory
    
    # ==========================================================================
    # Memory Fields (Redis-persisted)
    # ==========================================================================
    conversation_summary: str  # Summarized conversation history
    recent_messages: List[Dict[str, str]]  # Recent messages (sliding window)
    
    # ==========================================================================
    # Query Processing Fields
    # ==========================================================================
    intent: str  # Classified intent: chitchat, question, follow_up, tool
    needs_retrieval: bool  # Whether RAG retrieval is needed
    rewritten_query: str  # Query rewritten for better retrieval
    
    # ==========================================================================
    # Retrieval Fields
    # ==========================================================================
    retrieved_chunks: List[RetrievalHit]  # Retrieved document chunks
    
    # ==========================================================================
    # Tool Fields
    # ==========================================================================
    tool_name: Optional[str]  # Detected tool name
    tool_params: Dict[str, Any]  # Tool parameters
    tool_result: Optional[str]  # Tool execution result
    
    # ==========================================================================
    # Output Fields
    # ==========================================================================
    response: str  # Final generated response
    sources: List[Dict[str, Any]]  # Sources used in response


def create_initial_state(
    message: str,
    client_id: str,
    conversation_id: str,
    conversation_summary: str = "",
    recent_messages: Optional[List[Dict[str, str]]] = None,
    client_name: str = "",
) -> AgentState:
    """
    Create an initial agent state with default values.
    
    Args:
        message: User's input message
        client_id: Client ID for collection scoping
        conversation_id: Conversation ID for memory
        conversation_summary: Optional existing conversation summary
        recent_messages: Optional list of recent messages
        client_name: Optional client display name
        
    Returns:
        Initialized AgentState
    """
    return AgentState(
        # Input
        message=message,
        client_id=client_id,
        client_name=client_name or (client_id if client_id != "global" else "Global"),
        conversation_id=conversation_id,
        # Memory
        conversation_summary=conversation_summary,
        recent_messages=recent_messages or [],
        # Query processing
        intent="question",
        needs_retrieval=True,
        rewritten_query="",
        # Retrieval
        retrieved_chunks=[],
        # Tools
        tool_name=None,
        tool_params={},
        tool_result=None,
        # Output
        response="",
        sources=[],
    )
