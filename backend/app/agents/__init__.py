"""
Multi-agent LangGraph architecture for RAG chatbot.

This module provides a multi-agent system where specialized agents handle
different aspects of the conversation flow:

- QueryAgent: Intent classification and query rewriting
- RetrievalAgent: RAG retrieval from client + global collections
- SynthesisAgent: Response generation with sources
- ToolAgent: Calculator, DateTime, and extensible tools

The Orchestrator coordinates these agents using LangGraph.
"""

from app.agents.state import AgentState
from app.agents.orchestrator import get_orchestrator, Orchestrator

__all__ = [
    "AgentState",
    "Orchestrator",
    "get_orchestrator",
]
