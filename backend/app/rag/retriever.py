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

import json
import logging
import time
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

from app.config import settings
from app.models.schemas import RetrievalHit
from app.rag.vector_store import VectorStore, get_vector_store, get_client_vector_store
from app.rag.reranker import Reranker, MMRSelector, get_reranker, get_mmr_selector
from app.rag.hybrid_search import get_hybrid_search

logger = logging.getLogger(__name__)
_DEBUG_LOG_PATH = "/Users/g0y01hx/Desktop/personal_work/chatbot/.cursor/debug.log"


def _debug_log(payload: dict) -> None:
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload) + "\n")
    except Exception:
        pass


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
        from app.core.metrics import (
            record_retrieval_duration,
            record_retrieval_results,
            record_cross_client_filter,
            record_rerank_duration,
            record_rerank_skipped,
            record_stage_skipped,
            record_stage_executed,
        )
        
        start_time = time.time()
        fetch_k = fetch_k or settings.rag.initial_fetch_k
        lambda_mult = lambda_mult if lambda_mult is not None else settings.rag.mmr_lambda
        
        # Record that we're applying client filter
        record_cross_client_filter(client_id)
        
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
            record_retrieval_duration(client_id, "client_filtered", time.time() - start_time)
            record_retrieval_results(client_id, 0)
            return []
        
        logger.debug(f"Fetched {len(candidates)} candidates for client {client_id}")
        
        # Step 2: Confidence-based rerank gating
        should_skip_rerank = (
            skip_rerank_if_confident and 
            self._should_skip_rerank(candidates)
        )
        
        if self.reranker and not should_skip_rerank:
            record_stage_executed("rerank")
            rerank_start = time.time()
            rerank_k = min(len(candidates), top_k * 3)
            reranked = self.reranker.rerank(query, candidates, top_k=rerank_k)
            record_rerank_duration(time.time() - rerank_start)
            logger.debug(f"Reranked to {len(reranked)} candidates")
        else:
            if should_skip_rerank:
                record_rerank_skipped()
                record_stage_skipped("rerank", "high_confidence")
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
        
        # Record metrics
        record_retrieval_duration(client_id, "client_filtered", time.time() - start_time)
        record_retrieval_results(client_id, len(results))
        
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


def resolve_retrieval_scopes(
    client_id: Optional[str],
    allowed_clients: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Resolve which client scopes should be searched.

    Rules:
    - If client_id is None or "global": search global only.
    - If client_id is provided: search client + global (if client is allowed).
    """
    if not client_id or client_id == "global":
        return ["global"]

    if allowed_clients and client_id not in allowed_clients:
        logger.warning("Client not in allowed_clients; falling back to global scope")
        return ["global"]

    return [client_id, "global"]


def _merge_hits_by_scope(hits_by_scope: Dict[str, List[RetrievalHit]]) -> List[RetrievalHit]:
    """
    Merge retrieval hits from multiple scopes, de-duplicating by id.
    Prefers lower score (higher relevance). If equal, prefers non-global.
    """
    merged: Dict[str, RetrievalHit] = {}

    for scope, hits in hits_by_scope.items():
        for hit in hits:
            metadata = dict(hit.metadata) if hit.metadata else {}
            metadata.setdefault("scope", scope)
            if scope != "global":
                metadata.setdefault("client_id", scope)

            candidate = RetrievalHit(
                id=hit.id,
                content=hit.content,
                score=hit.score,
                metadata=metadata,
            )

            existing = merged.get(hit.id)
            if not existing:
                merged[hit.id] = candidate
                continue

            if candidate.score < existing.score:
                merged[hit.id] = candidate
                continue

            if candidate.score == existing.score:
                if candidate.metadata.get("scope") != "global" and existing.metadata.get("scope") == "global":
                    merged[hit.id] = candidate

    return list(merged.values())


def search_with_scopes(
    *,
    query: str,
    client_id: Optional[str],
    top_k: int,
    fetch_k: Optional[int] = None,
    metadata_filters: Optional[Dict] = None,
    allowed_clients: Optional[Iterable[str]] = None,
) -> List[RetrievalHit]:
    """
    Search across client + global scopes and merge results.
    """
    scopes = resolve_retrieval_scopes(client_id, allowed_clients)
    fetch_k = fetch_k or settings.rag.initial_fetch_k
    # #region agent log
    _debug_log({
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "H3",
        "location": "retriever.py:search_with_scopes",
        "message": "Search start",
        "data": {
            "client_id": client_id,
            "scopes": scopes,
            "top_k": top_k,
            "fetch_k": fetch_k,
            "allowed_clients": list(allowed_clients) if allowed_clients else None,
        },
        "timestamp": int(time.time() * 1000),
    })
    # #endregion agent log

    hits_by_scope: Dict[str, List[RetrievalHit]] = {}

    for scope in scopes:
        try:
            if settings.rag.bm25_enabled:
                hybrid = get_hybrid_search(client_id=scope)
                hits = hybrid.search(
                    query=query,
                    top_k=top_k,
                    fetch_k=fetch_k,
                    use_reranker=settings.rag.reranker_enabled,
                    where=metadata_filters,
                )
            else:
                retriever = get_retriever()
                hits = retriever.search_with_client_filter(
                    query=query,
                    client_id=scope,
                    top_k=top_k,
                    fetch_k=fetch_k,
                    metadata_filters=metadata_filters,
                )
        except Exception as e:
            logger.warning(f"Multi-scope retrieval failed for scope '{scope}': {e}")
            hits = []

        hits_by_scope[scope] = hits

    merged = _merge_hits_by_scope(hits_by_scope)
    merged.sort(key=lambda h: h.score)
    # #region agent log
    _debug_log({
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "H4",
        "location": "retriever.py:search_with_scopes",
        "message": "Search end",
        "data": {
            "counts_by_scope": {scope: len(hits) for scope, hits in hits_by_scope.items()},
            "merged_count": len(merged),
        },
        "timestamp": int(time.time() * 1000),
    })
    # #endregion agent log
    return merged[:top_k]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Get singleton retriever instance with reranking and MMR."""
    return Retriever(
        store=get_vector_store(),
        reranker=get_reranker(),
        mmr_selector=get_mmr_selector(),
    )

