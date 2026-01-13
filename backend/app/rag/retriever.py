"""
Retriever module with support for reranking and MMR diversity.

For small model optimization, the retrieval strategy is:
1. Fetch more candidates (top_k=30)
2. Rerank with cross-encoder
3. Apply MMR for diversity
4. Return top 3-5 high-quality, diverse results
"""

import logging
from functools import lru_cache
from typing import Dict, List, Optional

from app.config import settings
from app.models.schemas import RetrievalHit
from app.rag.vector_store import VectorStore, get_vector_store
from app.rag.reranker import Reranker, MMRSelector, get_reranker, get_mmr_selector

logger = logging.getLogger(__name__)


class Retriever:
    """
    Enhanced retriever with reranking and MMR support.
    
    Provides both simple search and advanced search_with_mmr for
    optimized retrieval with small models.
    """
    
    def __init__(
        self, 
        store: VectorStore,
        reranker: Optional[Reranker] = None,
        mmr_selector: Optional[MMRSelector] = None,
    ) -> None:
        self.store = store
        self.reranker = reranker
        self.mmr_selector = mmr_selector

    def search(
        self, query: str, top_k: int = 4, metadata_filters: Optional[Dict] = None
    ) -> List[RetrievalHit]:
        """
        Simple search without reranking or MMR.
        
        Use this for backward compatibility or when speed is critical.
        """
        return self.store.query(
            query=query, 
            top_k=top_k, 
            where=metadata_filters, 
            collection="documents"
        )
    
    def search_with_rerank(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: Optional[int] = None,
        metadata_filters: Optional[Dict] = None,
    ) -> List[RetrievalHit]:
        """
        Search with cross-encoder reranking.
        
        Args:
            query: Search query
            top_k: Final number of results to return
            fetch_k: Number of candidates to fetch for reranking (default: config)
            metadata_filters: Optional metadata filters
            
        Returns:
            Reranked list of RetrievalHit
        """
        fetch_k = fetch_k or settings.rag.initial_fetch_k
        
        # Fetch more candidates
        candidates = self.store.query(
            query=query,
            top_k=fetch_k,
            where=metadata_filters,
            collection="documents"
        )
        
        if not candidates:
            return []
        
        # Rerank if available
        if self.reranker:
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            logger.debug(f"Reranked {len(candidates)} -> {len(reranked)} results")
            return reranked
        
        # Fall back to simple truncation
        return candidates[:top_k]
    
    def search_with_mmr(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        metadata_filters: Optional[Dict] = None,
    ) -> List[RetrievalHit]:
        """
        Search with reranking and MMR diversity.
        
        This is the recommended method for small model RAG:
        1. Fetch many candidates (fetch_k)
        2. Rerank with cross-encoder
        3. Apply MMR for diversity
        4. Return top_k diverse, high-quality results
        
        Args:
            query: Search query
            top_k: Final number of results to return
            fetch_k: Candidates to fetch (default: config initial_fetch_k)
            lambda_mult: MMR lambda (default: config mmr_lambda)
            metadata_filters: Optional metadata filters
            
        Returns:
            Diverse, high-quality list of RetrievalHit
        """
        fetch_k = fetch_k or settings.rag.initial_fetch_k
        lambda_mult = lambda_mult if lambda_mult is not None else settings.rag.mmr_lambda
        
        # Step 1: Fetch candidates
        candidates = self.store.query(
            query=query,
            top_k=fetch_k,
            where=metadata_filters,
            collection="documents"
        )
        
        if not candidates:
            return []
        
        logger.debug(f"Fetched {len(candidates)} candidates for query: {query[:50]}...")
        
        # Step 2: Rerank with cross-encoder
        if self.reranker:
            # Rerank more than we need for MMR
            rerank_k = min(len(candidates), top_k * 3)  # Give MMR options
            reranked = self.reranker.rerank(query, candidates, top_k=rerank_k)
            logger.debug(f"Reranked to {len(reranked)} candidates")
        else:
            reranked = candidates
        
        # Step 3: Apply MMR for diversity
        if self.mmr_selector and len(reranked) > top_k:
            # Update lambda if different from selector default
            if lambda_mult != self.mmr_selector.lambda_mult:
                selector = MMRSelector(lambda_mult=lambda_mult)
            else:
                selector = self.mmr_selector
            
            diverse = selector.select(reranked, top_k=top_k)
            logger.debug(f"MMR selected {len(diverse)} diverse results")
            return diverse
        
        # Fall back to top_k
        return reranked[:top_k]
    
    def search_memories(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[RetrievalHit]:
        """Search memory collection (conversation summaries)."""
        return self.store.query(
            query=query,
            top_k=top_k,
            collection="memories"
        )


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Get singleton retriever instance with reranking and MMR."""
    return Retriever(
        store=get_vector_store(),
        reranker=get_reranker(),
        mmr_selector=get_mmr_selector(),
    )

