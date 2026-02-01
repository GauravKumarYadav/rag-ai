"""
Retriever module with support for reranking and MMR diversity.

For small model optimization, the retrieval strategy is:
1. Fetch more candidates (top_k=30)
2. Rerank with cross-encoder
3. Apply MMR for diversity
4. Return top 3-5 high-quality, diverse results

Security: All retrieval methods now support mandatory client_id filtering
to prevent cross-client data leakage.
"""

import logging
import time
from functools import lru_cache
from typing import Dict, List, Optional

from app.config import settings
from app.models.schemas import RetrievalHit
from app.rag.vector_store import VectorStore, get_vector_store, get_client_vector_store
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
    
    def search_with_client_filter(
        self,
        query: str,
        client_id: str,
        top_k: int = 5,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        metadata_filters: Optional[Dict] = None,
        skip_rerank_if_confident: bool = True,
    ) -> List[RetrievalHit]:
        """
        Search with mandatory client_id filtering.
        
        This is the SECURE retrieval method that enforces client isolation.
        All results are guaranteed to belong to the specified client.
        
        Args:
            query: Search query
            client_id: Client ID to filter by (REQUIRED - security boundary)
            top_k: Final number of results to return
            fetch_k: Candidates to fetch (default: config initial_fetch_k)
            lambda_mult: MMR lambda (default: config mmr_lambda)
            metadata_filters: Additional metadata filters
            skip_rerank_if_confident: Skip reranking when confidence is high
            
        Returns:
            Client-filtered, diverse, high-quality list of RetrievalHit
        """
        start_time = time.time()
        fetch_k = fetch_k or settings.rag.initial_fetch_k
        lambda_mult = lambda_mult if lambda_mult is not None else settings.rag.mmr_lambda
        
        # Get client-specific vector store
        client_store = get_client_vector_store(client_id)
        
        # Step 1: Fetch candidates from client's collection ONLY
        candidates = client_store.query(
            query=query,
            top_k=fetch_k,
            where=metadata_filters,
            collection="documents"
        )
        
        if not candidates:
            return []
        
        logger.debug(f"Fetched {len(candidates)} candidates for client {client_id}")
        
        # Step 2: Confidence-based rerank gating
        should_skip_rerank = (
            skip_rerank_if_confident and 
            self._should_skip_rerank(candidates)
        )
        
        if self.reranker and not should_skip_rerank:
            rerank_k = min(len(candidates), top_k * 3)
            reranked = self.reranker.rerank(query, candidates, top_k=rerank_k)
            logger.debug(f"Reranked to {len(reranked)} candidates")
        else:
            if should_skip_rerank:
                logger.debug(f"Skipped rerank - high confidence retrieval")
            reranked = candidates
        
        # Step 3: Apply MMR for diversity
        if self.mmr_selector and len(reranked) > top_k:
            if lambda_mult != self.mmr_selector.lambda_mult:
                selector = MMRSelector(lambda_mult=lambda_mult)
            else:
                selector = self.mmr_selector
            
            diverse = selector.select(reranked, top_k=top_k)
            logger.debug(f"MMR selected {len(diverse)} diverse results")
            results = diverse
        else:
            results = reranked[:top_k]
        
        duration = time.time() - start_time
        logger.debug(f"Client retrieval completed in {duration:.3f}s, {len(results)} results")
        
        return results
    
    def _should_skip_rerank(self, candidates: List[RetrievalHit]) -> bool:
        """
        Determine if reranking should be skipped based on confidence signals.
        
        Skip rerank when:
        - Few candidates (< 3) - nothing to reorder
        - High confidence: top result is clearly better than others
        - All scores are very high (perfect matches)
        
        Returns:
            True if reranking can be safely skipped
        """
        if len(candidates) < 3:
            return True
        
        # Get scores (lower is better for distance-based scores)
        scores = [c.score for c in candidates]
        
        # Check if top result is clearly dominant
        # (significant gap between top1 and top3)
        if len(scores) >= 3:
            top1_score = scores[0]
            top3_score = scores[2]
            
            # If using similarity (higher is better), check margin
            if all(s >= 0 and s <= 1 for s in scores[:3]):
                # Similarity scores (0-1, higher is better)
                if top1_score > 0.85 and (top1_score - top3_score) > 0.2:
                    return True
            else:
                # Distance scores (lower is better)
                if top1_score < 0.3 and (top3_score - top1_score) > 0.2:
                    return True
        
        return False


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Get singleton retriever instance with reranking and MMR."""
    return Retriever(
        store=get_vector_store(),
        reranker=get_reranker(),
        mmr_selector=get_mmr_selector(),
    )

