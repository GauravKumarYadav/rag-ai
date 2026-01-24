"""
Orchestrator for multi-agent LangGraph system.

This module provides the main LangGraph that coordinates the multi-agent system:
1. Query Agent - Intent classification and query rewriting
2. Retrieval Agent - RAG retrieval from client + global collections
3. Tool Agent - Tool execution (calculator, datetime)
4. Synthesis Agent - Response generation
5. Document List Handler - Lists available documents for a client

The orchestrator routes requests to appropriate agents based on the state.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph
from langsmith import traceable

from app.config import settings
from app.agents.state import AgentState, create_initial_state
from app.agents.query_agent import get_query_agent
from app.agents.retrieval_agent import get_retrieval_agent
from app.agents.synthesis_agent import get_synthesis_agent
from app.agents.tool_agent import get_tool_agent
from app.models.schemas import RetrievalHit
from app.rag.chroma_store import ChromaClientVectorStore, GlobalVectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestrator that coordinates the multi-agent system.
    
    Uses LangGraph to build a stateful workflow that routes requests
    through specialized agents based on intent and needs.
    """
    
    def __init__(self) -> None:
        self.query_agent = get_query_agent()
        self.retrieval_agent = get_retrieval_agent()
        self.synthesis_agent = get_synthesis_agent()
        self.tool_agent = get_tool_agent()
        self.max_steps = settings.agent.max_steps
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Flow:
        1. query_node: Classify intent, detect tools, rewrite query
        2. Route based on intent:
           - tool -> tool_node -> synthesis_node
           - question/follow_up -> retrieval_node -> synthesis_node
           - document_list -> document_list_node -> END (response set directly)
           - chitchat -> synthesis_node
        3. synthesis_node: Generate response
        4. END
        
        Returns:
            Compiled LangGraph
        """
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("query", self._query_node)
        graph.add_node("retrieval", self._retrieval_node)
        graph.add_node("tool", self._tool_node)
        graph.add_node("synthesis", self._synthesis_node)
        graph.add_node("document_list", self._document_list_node)
        
        # Set entry point
        graph.set_entry_point("query")
        
        # Add conditional routing after query
        graph.add_conditional_edges(
            "query",
            self._route_after_query,
            {
                "tool": "tool",
                "retrieval": "retrieval",
                "synthesis": "synthesis",
                "document_list": "document_list",
            },
        )
        
        # Tool -> Synthesis
        graph.add_edge("tool", "synthesis")
        
        # Retrieval -> Synthesis
        graph.add_edge("retrieval", "synthesis")
        
        # Document List -> END (response already set)
        graph.add_edge("document_list", END)
        
        # Synthesis -> END
        graph.add_edge("synthesis", END)
        
        return graph.compile()
    
    def _route_after_query(self, state: AgentState) -> str:
        """
        Route to the next node based on query analysis.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node name
        """
        # Check for tool usage
        if state.get("tool_name"):
            return "tool"
        
        # Check intent
        intent = state.get("intent", "question")
        
        if intent == "chitchat":
            return "synthesis"
        
        if intent == "document_list":
            return "document_list"
        
        if state.get("needs_retrieval", True):
            return "retrieval"
        
        return "synthesis"
    
    @traceable(name="agent.query")
    async def _query_node(self, state: AgentState) -> AgentState:
        """Query agent node - classifies intent and rewrites query."""
        return await self.query_agent.process(state)
    
    @traceable(name="agent.retrieval")
    async def _retrieval_node(self, state: AgentState) -> AgentState:
        """Retrieval agent node - fetches relevant documents."""
        return await self.retrieval_agent.process(state)
    
    @traceable(name="agent.tool")
    async def _tool_node(self, state: AgentState) -> AgentState:
        """Tool agent node - executes tools like calculator."""
        return await self.tool_agent.process(state)
    
    @traceable(name="agent.synthesis")
    async def _synthesis_node(self, state: AgentState) -> AgentState:
        """Synthesis agent node - generates final response."""
        return await self.synthesis_agent.process(state)
    
    @traceable(name="agent.document_list")
    async def _document_list_node(self, state: AgentState) -> AgentState:
        """
        Document list node - returns list of available documents for the client.
        
        Fetches document metadata from both client-specific and global collections.
        """
        client_id = state.get("client_id", "global")
        client_name = state.get("client_name", client_id)
        
        try:
            documents = await self._get_document_list(client_id)
            response = self._format_document_list(documents, client_name, client_id)
            return {**state, "response": response}
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return {**state, "response": "I encountered an error retrieving the document list. Please try again."}
    
    async def _get_document_list(self, client_id: str) -> List[Dict[str, Any]]:
        """
        Get list of documents for a client.
        
        Args:
            client_id: Client ID to get documents for
            
        Returns:
            List of document metadata dicts
        """
        documents = []
        seen_sources = set()  # Track unique source filenames
        
        # Get client-specific documents if not global
        if client_id and client_id != "global":
            try:
                config = VectorStoreConfig(
                    path=settings.rag.chroma_db_path,
                    url=settings.rag.url,
                    collection_prefix=settings.rag.collection_prefix,
                )
                client_store = ChromaClientVectorStore(config, client_id, verify_fingerprint=False)
                client_docs = client_store.docs.get(include=["metadatas"])
                
                if client_docs and client_docs.get("metadatas"):
                    for meta in client_docs["metadatas"]:
                        if meta:
                            source = meta.get("source") or meta.get("source_filename", "Unknown")
                            if source not in seen_sources:
                                seen_sources.add(source)
                                documents.append({
                                    "source": source,
                                    "client_id": client_id,
                                    "is_global": False,
                                    "uploaded_at": meta.get("uploaded_at", meta.get("created_at", "")),
                                    "metadata": meta,
                                })
            except Exception as e:
                logger.warning(f"Failed to get client documents for {client_id}: {e}")
        
        # Get global documents (create fresh instance, no caching)
        try:
            config = VectorStoreConfig(
                path=settings.rag.chroma_db_path,
                url=settings.rag.url,
                collection_prefix=settings.rag.collection_prefix,
            )
            global_store = GlobalVectorStore(config)
            global_docs = global_store.docs.get(include=["metadatas"])
            
            if global_docs and global_docs.get("metadatas"):
                for meta in global_docs["metadatas"]:
                    if meta:
                        source = meta.get("source") or meta.get("source_filename", "Unknown")
                        global_source_key = f"global:{source}"
                        if global_source_key not in seen_sources:
                            seen_sources.add(global_source_key)
                            documents.append({
                                "source": source,
                                "client_id": "global",
                                "is_global": True,
                                "uploaded_at": meta.get("uploaded_at", meta.get("created_at", "")),
                                "metadata": meta,
                            })
        except Exception as e:
            logger.warning(f"Failed to get global documents: {e}")
        
        return documents
    
    def _format_document_list(self, documents: List[Dict[str, Any]], client_name: str, client_id: str) -> str:
        """
        Format document list as a human-readable response.
        
        Args:
            documents: List of document metadata
            client_name: Client display name
            client_id: Client ID
            
        Returns:
            Formatted response string
        """
        if not documents:
            if client_id == "global":
                return "There are no documents in the knowledge base yet."
            else:
                return f"There are no documents in {client_name}'s knowledge base yet."
        
        # Separate client and global documents
        client_docs = [d for d in documents if not d.get("is_global")]
        global_docs = [d for d in documents if d.get("is_global")]
        
        response_parts = []
        
        # Format client documents
        if client_docs:
            if client_id != "global":
                response_parts.append(f"**Documents in {client_name}'s knowledge base:**")
            else:
                response_parts.append("**Documents in the knowledge base:**")
            
            for i, doc in enumerate(client_docs, 1):
                source = doc["source"]
                uploaded = doc.get("uploaded_at", "")
                if uploaded:
                    try:
                        # Try to parse and format the date
                        if isinstance(uploaded, str) and uploaded:
                            dt = datetime.fromisoformat(uploaded.replace("Z", "+00:00"))
                            uploaded = dt.strftime("%b %d, %Y")
                            response_parts.append(f"{i}. {source} (uploaded {uploaded})")
                        else:
                            response_parts.append(f"{i}. {source}")
                    except Exception:
                        response_parts.append(f"{i}. {source}")
                else:
                    response_parts.append(f"{i}. {source}")
        
        # Format global documents
        if global_docs and client_id != "global":
            if response_parts:
                response_parts.append("")  # Empty line separator
            response_parts.append("**Shared global documents:**")
            
            for i, doc in enumerate(global_docs, 1):
                source = doc["source"]
                response_parts.append(f"{i}. {source}")
        
        return "\n".join(response_parts)
    
    @traceable(name="orchestrator.run")
    async def run(
        self,
        message: str,
        client_id: str,
        conversation_id: str = "",
        conversation_summary: str = "",
        recent_messages: Optional[List[dict]] = None,
        client_name: str = "",
    ) -> Tuple[str, List[RetrievalHit]]:
        """
        Run the multi-agent workflow.
        
        Args:
            message: User's input message
            client_id: Client ID for collection scoping
            conversation_id: Conversation ID for memory
            conversation_summary: Optional conversation summary
            recent_messages: Optional list of recent messages
            client_name: Client display name for context
            
        Returns:
            Tuple of (response, retrieved_chunks)
        """
        # Create initial state
        state = create_initial_state(
            message=message,
            client_id=client_id,
            conversation_id=conversation_id,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
            client_name=client_name,
        )
        
        try:
            # Run the graph
            result = await self.graph.ainvoke(state)
            
            response = result.get("response", "")
            retrieved = result.get("retrieved_chunks", [])
            
            return response, retrieved
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return "I encountered an error processing your request. Please try again.", []


# Singleton instance
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get or create the Orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


# Backwards compatibility alias
def get_langgraph_agent() -> Orchestrator:
    """
    Backwards compatibility alias for get_orchestrator.
    
    Returns:
        Orchestrator instance
    """
    return get_orchestrator()


# Keep old class name for backwards compatibility
LangGraphAgent = Orchestrator
