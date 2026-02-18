"""
Cross-Encoder Reranker for Small Model RAG Optimization.

Uses sentence-transformers CrossEncoder models to rerank retrieval results.
Cross-encoders are more accurate than bi-encoders for reranking because they
jointly encode query and document, but they're slower (hence used post-retrieval).

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Small and fast (~22M params)
- Trained on MS MARCO passage ranking
- Good balance of speed and accuracy

Concurrency: A semaphore limits concurrent reranker runs so multiple users
do not saturate CPU; accuracy is unchanged.
"""

import asyncio
import logging
from functools import lru_cache
from typing import List, Optional

import numpy as np

from app.config import settings
from app.models.schemas import RetrievalHit

logger = logging.getLogger(__name__)

# Initialize semaphore once at module level based on settings
# Use settings.rag.reranker_max_concurrent if available, default to 2
_max_concurrent = getattr(settings.rag, 'reranker_max_concurrent', 2)
RERANKER_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(_max_concurrent)


def get_reranker_semaphore() -> asyncio.Semaphore:
    """
    Return the global reranker semaphore.
    
    The semaphore is initialized once at module load time based on
    settings.rag.reranker_max_concurrent (default: 2).
    """
    return RERANKER_SEMAPHORE


def reload_reranker_semaphore(max_concurrent: int) -> None:
    """
    Reload the reranker semaphore with a new max concurrent value.
    
    Note: This is primarily for testing or hot-reload scenarios.
    Existing waiters on the old semaphore will not be affected.
    
    Args:
        max_concurrent: New maximum concurrent reranker operations
    """
    global RERANKER_SEMAPHORE
    RERANKER_SEMAPHORE = asyncio.Semaphore(max_concurrent)
    logger.info(f"Reranker semaphore reloaded with max_concurrent={max_concurrent}")


# Lazy load sentence-transformers to avoid startup overhead
_cross_encoder = None


def _get_cross_encoder():
    """Lazy load the CrossEncoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            model_name = settings.rag.reranker_model
            logger.info(f"Loading CrossEncoder model: {model_name}")
            _cross_encoder = CrossEncoder(model_name)
            logger.info("CrossEncoder model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed. Reranking disabled.")
            _cross_encoder = False
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}")
            _cross_encoder = False
    return _cross_encoder if _cross_encoder else None


class Reranker:
    """
    Cross-encoder reranker for improving retrieval quality.
    
    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, hits, top_k=5)
    """
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialize the reranker.
        
        Args:
            model_name: CrossEncoder model name. If None, uses config setting.
        """
        self.model_name = model_name or settings.rag.reranker_model
        self.enabled = settings.rag.reranker_enabled
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None and self.enabled:
            self._model = _get_cross_encoder()
        return self._model
    
    def rerank(
        self, 
        query: str, 
        hits: List[RetrievalHit], 
        top_k: Optional[int] = None
    ) -> List[RetrievalHit]:
        """
        Rerank retrieval hits using cross-encoder scoring.
        
        Args:
            query: The search query
            hits: List of RetrievalHit to rerank
            top_k: Number of top results to return (None = all)
            
        Returns:
            Reranked list of RetrievalHit, sorted by rerank score (descending)
        """
        if not hits:
            return []
        
        if not self.enabled or self.model is None:
            logger.debug("Reranker disabled or unavailable, returning original order")
            return hits[:top_k] if top_k else hits
        
        try:
            # Create query-document pairs for scoring
            pairs = [(query, hit.content) for hit in hits]
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Combine hits with new scores
            scored_hits = []
            for hit, rerank_score in zip(hits, scores):
                # Create new hit with rerank score
                # Store original score in metadata for reference
                new_hit = RetrievalHit(
                    id=hit.id,
                    content=hit.content,
                    score=float(rerank_score),  # Replace with rerank score
                    metadata={
                        **hit.metadata,
                        "original_score": hit.score,
                        "rerank_score": float(rerank_score),
                    }
                )
                scored_hits.append(new_hit)
            
            # Sort by rerank score (higher is better for cross-encoders)
            scored_hits.sort(key=lambda x: x.score, reverse=True)
            
            # Return top_k if specified
            if top_k:
                scored_hits = scored_hits[:top_k]
            
            logger.debug(f"Reranked {len(hits)} hits, returning top {len(scored_hits)}")
            return scored_hits
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original order")
            return hits[:top_k] if top_k else hits
    
    def compute_scores(self, query: str, hits: List[RetrievalHit]) -> List[float]:
        """
        Compute rerank scores without creating new hits.
        
        Useful when you need scores separately from reranking.
        
        Returns:
            List of scores in same order as input hits
        """
        if not hits:
            return []
        
        if not self.enabled or self.model is None:
            # Return normalized original scores
            return [hit.score for hit in hits]
        
        try:
            pairs = [(query, hit.content) for hit in hits]
            scores = self.model.predict(pairs)
            return [float(s) for s in scores]
        except Exception as e:
            logger.error(f"Score computation failed: {e}")
            return [hit.score for hit in hits]


class MMRSelector:
    """
    Maximal Marginal Relevance (MMR) selector for result diversification.
    
    MMR balances relevance with diversity by penalizing documents that are
    too similar to already-selected documents.
    
    Formula: MMR = λ * sim(d, q) - (1-λ) * max(sim(d, d_i))
    where d_i are already selected documents.
    """
    
    def __init__(self, lambda_mult: float = 0.5) -> None:
        """
        Initialize MMR selector.
        
        Args:
            lambda_mult: Balance between relevance and diversity.
                        1.0 = pure relevance (no diversity)
                        0.0 = pure diversity (ignore relevance)
                        0.5 = balanced (default)
        """
        self.lambda_mult = lambda_mult
    
    def select(
        self,
        hits: List[RetrievalHit],
        top_k: int,
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[RetrievalHit]:
        """
        Select top_k documents using MMR.
        
        Args:
            hits: Candidate hits (should be pre-sorted by relevance)
            top_k: Number of documents to select
            embeddings: Optional pre-computed embeddings for diversity calculation.
                       If None, uses simple text overlap for diversity.
                       
        Returns:
            Selected hits with diversity
        """
        if not hits or top_k <= 0:
            return []
        
        if len(hits) <= top_k:
            return hits
        
        # If no embeddings, use text-based similarity
        if embeddings is None:
            return self._select_with_text_similarity(hits, top_k)
        
        return self._select_with_embeddings(hits, top_k, embeddings)
    
    def _select_with_text_similarity(
        self,
        hits: List[RetrievalHit],
        top_k: int,
    ) -> List[RetrievalHit]:
        """
        MMR selection using simple text overlap for diversity.
        
        Uses Jaccard similarity on word sets as a proxy for semantic similarity.
        """
        selected: List[RetrievalHit] = []
        candidates = list(hits)
        
        # Precompute word sets for each document
        word_sets = [set(hit.content.lower().split()) for hit in candidates]
        
        while len(selected) < top_k and candidates:
            best_idx = 0
            best_score = float("-inf")
            
            for idx, candidate in enumerate(candidates):
                # Relevance score (from reranking or original)
                relevance = candidate.score
                
                # Diversity: max similarity to already selected docs
                if selected:
                    max_sim = 0.0
                    cand_words = word_sets[idx]
                    for sel_hit in selected:
                        sel_words = set(sel_hit.content.lower().split())
                        # Jaccard similarity
                        intersection = len(cand_words & sel_words)
                        union = len(cand_words | sel_words)
                        sim = intersection / union if union > 0 else 0
                        max_sim = max(max_sim, sim)
                else:
                    max_sim = 0.0
                
                # MMR score
                mmr_score = self.lambda_mult * relevance - (1 - self.lambda_mult) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            # Add best candidate to selected
            selected.append(candidates[best_idx])
            # Remove from candidates
            candidates.pop(best_idx)
            word_sets.pop(best_idx)
        
        return selected
    
    def _select_with_embeddings(
        self,
        hits: List[RetrievalHit],
        top_k: int,
        embeddings: List[List[float]],
    ) -> List[RetrievalHit]:
        """
        MMR selection using embedding cosine similarity.
        """
        selected: List[RetrievalHit] = []
        selected_embeddings: List[np.ndarray] = []
        candidates = list(zip(hits, embeddings))
        
        while len(selected) < top_k and candidates:
            best_idx = 0
            best_score = float("-inf")
            
            for idx, (candidate, emb) in enumerate(candidates):
                emb_array = np.array(emb)
                
                # Relevance score
                relevance = candidate.score
                
                # Diversity: max cosine similarity to selected
                if selected_embeddings:
                    similarities = [
                        self._cosine_similarity(emb_array, sel_emb)
                        for sel_emb in selected_embeddings
                    ]
                    max_sim = max(similarities)
                else:
                    max_sim = 0.0
                
                # MMR score
                mmr_score = self.lambda_mult * relevance - (1 - self.lambda_mult) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            # Add best candidate
            hit, emb = candidates[best_idx]
            selected.append(hit)
            selected_embeddings.append(np.array(emb))
            candidates.pop(best_idx)
        
        return selected
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Get singleton reranker instance."""
    return Reranker()


@lru_cache(maxsize=1)
def get_mmr_selector() -> MMRSelector:
    """Get singleton MMR selector instance."""
    return MMRSelector(lambda_mult=settings.rag.mmr_lambda)
