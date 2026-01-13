"""
Hybrid Search with Reciprocal Rank Fusion (RRF).

Combines BM25 keyword search with vector semantic search for better retrieval.

Key benefits:
- BM25 excels at exact keyword matches and factual lookups
- Vector search excels at semantic similarity and paraphrasing
- RRF fusion combines both without requiring score calibration

Security:
- All searches are scoped to a specific client_id
- Cross-client data leakage is prevented by design

References:
- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). 
  Reciprocal rank fusion outperforms condorcet and individual rank learning methods.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config import settings
from app.models.schemas import RetrievalHit
from app.rag.base import VectorStoreBase
from app.rag.bm25_index import BM25Index, get_bm25_index
from app.rag.reranker import Reranker, get_reranker

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    hits: List[RetrievalHit]
    bm25_candidates: int
    vector_candidates: int
    fused_candidates: int


class HybridSearch:
    """
    Hybrid search combining BM25 and vector retrieval with RRF fusion.
    
    Pipeline:
    1. Query both BM25 and vector indices (client-scoped)
    2. Apply Reciprocal Rank Fusion to combine results
    3. Optionally rerank with cross-encoder
    4. Return top-k results
    
    Security: All searches are scoped to the client_id provided at initialization.
    """
    
    def __init__(
        self,
        vector_store: VectorStoreBase,
        bm25_index: BM25Index,
        reranker: Optional[Reranker] = None,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        rrf_k: int = 60,
        client_id: Optional[str] = None,
    ):
        """
        Initialize hybrid search.
        
        Args:
            vector_store: Vector database for semantic search
            bm25_index: BM25 index for keyword search
            reranker: Optional cross-encoder reranker
            bm25_weight: Weight for BM25 in fusion (0-1)
            vector_weight: Weight for vector in fusion (0-1)
            rrf_k: RRF constant (higher = smoother fusion)
            client_id: Client ID this search is scoped to
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.client_id = client_id or "global"
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 30,
        where: Optional[Dict] = None,
        collection: str = "documents",
        use_reranker: bool = True,
    ) -> List[RetrievalHit]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            fetch_k: Number of candidates to fetch from each source
            where: Metadata filter
            collection: Collection to search
            use_reranker: Whether to apply cross-encoder reranking
            
        Returns:
            List of RetrievalHit sorted by relevance
        """
        result = self.search_with_stats(
            query=query,
            top_k=top_k,
            fetch_k=fetch_k,
            where=where,
            collection=collection,
            use_reranker=use_reranker,
        )
        return result.hits
    
    def search_with_stats(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 30,
        where: Optional[Dict] = None,
        collection: str = "documents",
        use_reranker: bool = True,
    ) -> HybridSearchResult:
        """
        Perform hybrid search with detailed statistics.
        
        All results are filtered to the client_id this HybridSearch was created for.
        
        Returns:
            HybridSearchResult with hits and stats
        """
        from app.core.metrics import (
            record_retrieval_duration,
            record_retrieval_results,
            record_cross_client_filter,
            record_rerank_duration,
        )
        
        start_time = time.time()
        
        # Record that we're applying client filter
        record_cross_client_filter(self.client_id)
        
        # 1. Get candidates from vector search (client-scoped)
        vector_hits = self.vector_store.query(
            query=query,
            top_k=fetch_k,
            where=where,
            collection=collection,
        )
        logger.debug(f"Vector search for client {self.client_id} returned {len(vector_hits)} candidates")
        
        # 2. Get candidates from BM25 search (client-scoped)
        bm25_hits = self.bm25_index.search(
            query=query,
            top_k=fetch_k,
            where=where,
        )
        logger.debug(f"BM25 search for client {self.client_id} returned {len(bm25_hits)} candidates")
        
        # 3. Apply Reciprocal Rank Fusion
        fused_hits = self._rrf_fusion(vector_hits, bm25_hits)
        logger.debug(f"RRF fusion produced {len(fused_hits)} candidates")
        
        # 4. Optionally rerank with cross-encoder
        if use_reranker and self.reranker and fused_hits:
            rerank_start = time.time()
            fused_hits = self.reranker.rerank(query, fused_hits, top_k=top_k)
            record_rerank_duration(time.time() - rerank_start)
            logger.debug(f"Reranker returned {len(fused_hits)} results")
        else:
            fused_hits = fused_hits[:top_k]
        
        # Record metrics
        record_retrieval_duration(self.client_id, "hybrid", time.time() - start_time)
        record_retrieval_results(self.client_id, len(fused_hits))
        
        return HybridSearchResult(
            hits=fused_hits,
            bm25_candidates=len(bm25_hits),
            vector_candidates=len(vector_hits),
            fused_candidates=len(fused_hits),
        )
    
    def _rrf_fusion(
        self,
        vector_hits: List[RetrievalHit],
        bm25_hits: List[RetrievalHit],
    ) -> List[RetrievalHit]:
        """
        Apply Reciprocal Rank Fusion to combine two ranked lists.
        
        RRF score = sum(weight / (k + rank)) for each list
        
        Args:
            vector_hits: Results from vector search
            bm25_hits: Results from BM25 search
            
        Returns:
            Fused and sorted list of hits
        """
        # Build document lookup
        doc_map: Dict[str, RetrievalHit] = {}
        scores: Dict[str, float] = {}
        
        # Score from vector results
        for rank, hit in enumerate(vector_hits):
            rrf_score = self.vector_weight / (self.rrf_k + rank + 1)
            scores[hit.id] = scores.get(hit.id, 0) + rrf_score
            doc_map[hit.id] = hit
        
        # Score from BM25 results
        for rank, hit in enumerate(bm25_hits):
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)
            scores[hit.id] = scores.get(hit.id, 0) + rrf_score
            if hit.id not in doc_map:
                doc_map[hit.id] = hit
        
        # Sort by fused score (higher is better)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Create result list with updated scores
        results = []
        for doc_id in sorted_ids:
            hit = doc_map[doc_id]
            # Update score to RRF score (normalized to 0-1)
            max_possible = self.vector_weight / (self.rrf_k + 1) + self.bm25_weight / (self.rrf_k + 1)
            normalized_score = scores[doc_id] / max_possible
            
            results.append(RetrievalHit(
                id=hit.id,
                content=hit.content,
                score=1.0 - normalized_score,  # Convert to distance (lower is better)
                metadata=hit.metadata,
            ))
        
        return results
    
    def add_documents(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        """
        Add documents to both indices.
        
        This should be called when new documents are ingested to keep
        both indices in sync.
        """
        # Add to BM25 index
        self.bm25_index.add_documents(contents, ids, metadatas)
        
        # Vector store is typically updated separately via its own interface
        logger.debug(f"Added {len(contents)} documents to BM25 index")
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
    ) -> int:
        """
        Delete documents from BM25 index.
        
        Vector store deletion should be handled separately.
        """
        return self.bm25_index.delete(ids=ids, where=where)


# Global hybrid search instances
_hybrid_search_instances: Dict[str, HybridSearch] = {}


def get_hybrid_search(
    client_id: Optional[str] = None,
    vector_store: Optional[VectorStoreBase] = None,
) -> HybridSearch:
    """
    Get or create a hybrid search instance for a specific client.
    
    All searches using this instance will be scoped to the client_id.
    This ensures cross-client data isolation.
    
    Args:
        client_id: Client ID for per-client search (defaults to 'global')
        vector_store: Vector store to use (optional override)
        
    Returns:
        HybridSearch instance scoped to the client
    """
    from app.rag.vector_store import get_vector_store, get_client_vector_store
    
    # Always use a client_id (default to 'global')
    effective_client_id = client_id or "global"
    cache_key = effective_client_id
    
    if cache_key not in _hybrid_search_instances:
        # Get or create components for this client
        vs = vector_store or get_client_vector_store(effective_client_id)
        bm25 = get_bm25_index(client_id=effective_client_id)
        reranker = get_reranker() if settings.rag.reranker_enabled else None
        
        _hybrid_search_instances[cache_key] = HybridSearch(
            vector_store=vs,
            bm25_index=bm25,
            reranker=reranker,
            bm25_weight=getattr(settings.rag, 'bm25_weight', 0.4),
            vector_weight=getattr(settings.rag, 'vector_weight', 0.6),
            client_id=effective_client_id,
        )
    
    return _hybrid_search_instances[cache_key]


def clear_hybrid_search_cache() -> None:
    """Clear the hybrid search cache."""
    global _hybrid_search_instances
    _hybrid_search_instances = {}
