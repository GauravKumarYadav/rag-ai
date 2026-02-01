"""
Retrieval Agent for RAG retrieval from client + global collections.

This agent is responsible for:
1. Searching the client-specific collection
2. Searching the global collection
3. Merging and deduplicating results
4. Reranking combined results

Hybrid search (BM25 + reranker) runs under a semaphore and in a thread pool
so concurrent users do not saturate CPU; accuracy is unchanged.
"""

import asyncio
import logging
from typing import List, Optional

from langsmith import traceable

from app.config import settings
from app.agents.state import AgentState
from app.models.schemas import RetrievalHit
from app.rag.hybrid_search import get_hybrid_search
from app.rag.reranker import get_reranker_semaphore
from app.rag.retriever import get_retriever

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Agent responsible for RAG retrieval from multiple collections.
    """
    
    def __init__(self) -> None:
        self.top_k = settings.rag.rerank_top_k
        self.fetch_k = settings.rag.initial_fetch_k
        self.use_reranker = settings.rag.reranker_enabled
        self.use_bm25 = settings.rag.bm25_enabled
    
    @traceable(name="retrieval_agent.process")
    async def process(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant documents from client and global collections.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with retrieved chunks
        """
        # Skip retrieval if not needed
        if not state.get("needs_retrieval", True):
            return state
        
        client_id = state.get("client_id") or "global"
        query = state.get("rewritten_query") or state.get("message", "")
        
        if not query:
            return state
        
        try:
            # Retrieve from client collection + global collection
            retrieved = await self._retrieve_with_global(query, client_id)
            
            # Build sources list for response
            sources = self._build_sources(retrieved)
            
            return {
                **state,
                "retrieved_chunks": retrieved,
                "sources": sources,
            }
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                **state,
                "retrieved_chunks": [],
                "sources": [],
            }
    
    @traceable(name="retrieval_agent.retrieve_with_global")
    async def _retrieve_with_global(
        self,
        query: str,
        client_id: str,
    ) -> List[RetrievalHit]:
        """
        Retrieve from both client-specific and global collections.
        
        Args:
            query: Search query
            client_id: Client ID for collection scoping
            
        Returns:
            Merged and deduplicated results
        """
        client_results: List[RetrievalHit] = []
        global_results: List[RetrievalHit] = []
        
        # Retrieve from client collection
        if client_id and client_id != "global":
            client_results = await self._search_collection(query, client_id)
            logger.debug(f"Retrieved {len(client_results)} from client '{client_id}'")
        
        # Retrieve from global collection
        global_results = await self._search_collection(query, "global")
        logger.debug(f"Retrieved {len(global_results)} from global collection")
        
        # Merge and deduplicate
        merged = self._merge_results(client_results, global_results)
        logger.debug(f"Merged to {len(merged)} results after deduplication")
        
        return merged[:self.top_k]
    
    async def _search_collection(
        self,
        query: str,
        client_id: str,
    ) -> List[RetrievalHit]:
        """
        Search a single collection.
        
        Args:
            query: Search query
            client_id: Client ID (or "global" for global collection)
            
        Returns:
            List of retrieval hits
        """
        try:
            if self.use_bm25:
                # Use hybrid search (BM25 + vector); limit concurrent reranker runs
                hybrid = get_hybrid_search(client_id=client_id)
                loop = asyncio.get_event_loop()
                async with get_reranker_semaphore():
                    return await loop.run_in_executor(
                        None,
                        lambda: hybrid.search(
                            query=query,
                            top_k=self.top_k,
                            fetch_k=self.fetch_k,
                            use_reranker=self.use_reranker,
                        ),
                    )
            else:
                # Use vector-only search
                retriever = get_retriever()
                return retriever.search_with_client_filter(
                    query=query,
                    client_id=client_id,
                    top_k=self.top_k,
                    fetch_k=self.fetch_k,
                )
        except Exception as e:
            logger.warning(f"Search failed for client '{client_id}': {e}")
            return []
    
    def _merge_results(
        self,
        client_results: List[RetrievalHit],
        global_results: List[RetrievalHit],
    ) -> List[RetrievalHit]:
        """
        Merge results from multiple collections, removing duplicates.
        
        Client results are prioritized over global results.
        Duplicates are detected by chunk ID.
        
        Args:
            client_results: Results from client collection
            global_results: Results from global collection
            
        Returns:
            Merged and deduplicated list
        """
        seen_ids = set()
        merged = []
        
        # Add client results first (priority)
        for hit in client_results:
            if hit.id not in seen_ids:
                seen_ids.add(hit.id)
                # Mark as client-specific
                hit.metadata["collection_type"] = "client"
                merged.append(hit)
        
        # Add global results
        for hit in global_results:
            if hit.id not in seen_ids:
                seen_ids.add(hit.id)
                # Mark as global
                hit.metadata["collection_type"] = "global"
                merged.append(hit)
        
        # Sort by score (lower is better for distance-based scoring)
        # If using similarity scores (higher is better), reverse the sort
        merged.sort(key=lambda x: x.score if x.score else float('inf'))
        
        return merged
    
    def _build_sources(self, retrieved: List[RetrievalHit]) -> List[dict]:
        """
        Build a list of sources from retrieved chunks.
        
        Args:
            retrieved: List of retrieval hits
            
        Returns:
            List of source dictionaries
        """
        sources = []
        for hit in retrieved:
            source = {
                "id": hit.id,
                "content_preview": hit.content[:200] + "..." if len(hit.content) > 200 else hit.content,
                "score": hit.score,
                "source": hit.metadata.get("source", hit.metadata.get("source_filename", "unknown")),
                "collection_type": hit.metadata.get("collection_type", "unknown"),
            }
            
            # Add optional metadata
            if "page_number" in hit.metadata:
                source["page"] = hit.metadata["page_number"]
            if "section_heading" in hit.metadata:
                source["section"] = hit.metadata["section_heading"]
            
            sources.append(source)
        
        return sources


# Singleton instance
_retrieval_agent: Optional[RetrievalAgent] = None


def get_retrieval_agent() -> RetrievalAgent:
    """Get or create the RetrievalAgent singleton."""
    global _retrieval_agent
    if _retrieval_agent is None:
        _retrieval_agent = RetrievalAgent()
    return _retrieval_agent
