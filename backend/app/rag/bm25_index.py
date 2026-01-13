"""
BM25 Index for Keyword-based Search.

BM25 (Best Matching 25) is a probabilistic retrieval model that ranks documents
based on term frequency and inverse document frequency. It excels at:
- Exact keyword matches
- Factual lookups with specific terms
- Queries with rare/unique terms

This complements vector search which is better for semantic similarity.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from rank_bm25 import BM25Okapi

from app.models.schemas import RetrievalHit

logger = logging.getLogger(__name__)


@dataclass
class BM25Document:
    """A document indexed in BM25."""
    id: str
    content: str
    tokens: List[str]
    metadata: Dict = field(default_factory=dict)


class BM25Index:
    """
    BM25 index for keyword-based document retrieval.
    
    Maintains an in-memory BM25 index that can be persisted to disk.
    Supports per-client isolation through separate indices.
    """
    
    # Common English stopwords to filter
    STOPWORDS: Set[str] = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "the", "this", "but", "they",
        "have", "had", "what", "when", "where", "who", "which", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "can", "should", "now", "i", "you",
        "your", "we", "our", "their", "them", "his", "her", "she", "him",
    }
    
    def __init__(
        self,
        persist_path: Optional[str] = None,
        client_id: Optional[str] = None,
    ):
        """
        Initialize BM25 index.
        
        Args:
            persist_path: Directory to persist index data
            client_id: Optional client ID for per-client isolation
        """
        self.client_id = client_id
        self.persist_path = Path(persist_path) if persist_path else None
        
        # Document storage
        self.documents: Dict[str, BM25Document] = {}
        self.doc_order: List[str] = []  # Maintains order for BM25 alignment
        
        # BM25 index (rebuilt when documents change)
        self._bm25: Optional[BM25Okapi] = None
        self._dirty = False
        
        # Load from disk if available
        if self.persist_path:
            self._load_from_disk()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Performs:
        - Lowercase conversion
        - Punctuation removal
        - Stopword filtering
        - Minimum token length filtering
        """
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        
        # Filter stopwords and short tokens
        tokens = [
            t for t in tokens
            if t not in self.STOPWORDS and len(t) > 2
        ]
        
        return tokens
    
    def add_documents(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        """
        Add documents to the BM25 index.
        
        Args:
            contents: List of document texts
            ids: List of document IDs (must be unique)
            metadatas: Optional list of metadata dicts
        """
        if metadatas is None:
            metadatas = [{} for _ in contents]
        
        for content, doc_id, meta in zip(contents, ids, metadatas):
            tokens = self._tokenize(content)
            
            doc = BM25Document(
                id=doc_id,
                content=content,
                tokens=tokens,
                metadata=meta,
            )
            
            # Update or add document
            if doc_id not in self.documents:
                self.doc_order.append(doc_id)
            
            self.documents[doc_id] = doc
        
        self._dirty = True
        logger.debug(f"Added {len(contents)} documents to BM25 index")
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
    ) -> int:
        """
        Delete documents from the index.
        
        Args:
            ids: Specific document IDs to delete
            where: Metadata filter for deletion
            
        Returns:
            Number of documents deleted
        """
        deleted = 0
        
        if ids:
            for doc_id in ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                    self.doc_order.remove(doc_id)
                    deleted += 1
        
        elif where:
            # Filter by metadata
            to_delete = []
            for doc_id, doc in self.documents.items():
                match = all(
                    doc.metadata.get(k) == v
                    for k, v in where.items()
                )
                if match:
                    to_delete.append(doc_id)
            
            for doc_id in to_delete:
                del self.documents[doc_id]
                self.doc_order.remove(doc_id)
                deleted += 1
        
        if deleted > 0:
            self._dirty = True
            logger.debug(f"Deleted {deleted} documents from BM25 index")
        
        return deleted
    
    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from current documents."""
        if not self.documents:
            self._bm25 = None
            return
        
        # Build corpus in document order
        corpus = [self.documents[doc_id].tokens for doc_id in self.doc_order]
        
        self._bm25 = BM25Okapi(corpus)
        self._dirty = False
        logger.debug(f"Rebuilt BM25 index with {len(corpus)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[RetrievalHit]:
        """
        Search the BM25 index.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            where: Optional metadata filter
            
        Returns:
            List of RetrievalHit objects sorted by relevance
        """
        if not self.documents:
            return []
        
        # Rebuild index if dirty
        if self._dirty or self._bm25 is None:
            self._rebuild_index()
        
        if self._bm25 is None:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)
        
        # Create scored results
        results: List[tuple] = []
        for idx, doc_id in enumerate(self.doc_order):
            doc = self.documents[doc_id]
            score = scores[idx]
            
            # Apply metadata filter if provided
            if where:
                match = all(
                    doc.metadata.get(k) == v
                    for k, v in where.items()
                )
                if not match:
                    continue
            
            results.append((doc, score))
        
        # Sort by score (higher is better for BM25)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to RetrievalHit
        hits = []
        for doc, score in results[:top_k]:
            # Normalize score to 0-1 range (BM25 scores can vary widely)
            # Lower score is better for consistency with distance metrics
            normalized_score = 1.0 / (1.0 + score) if score > 0 else 1.0
            
            hits.append(RetrievalHit(
                id=doc.id,
                content=doc.content,
                score=normalized_score,
                metadata=doc.metadata,
            ))
        
        return hits
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            "document_count": len(self.documents),
            "client_id": self.client_id,
            "index_built": self._bm25 is not None,
        }
    
    def _get_persist_file(self) -> Optional[Path]:
        """Get the persistence file path."""
        if not self.persist_path:
            return None
        
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"bm25_{self.client_id}.json" if self.client_id else "bm25_global.json"
        return self.persist_path / filename
    
    def persist(self) -> None:
        """Persist index to disk."""
        file_path = self._get_persist_file()
        if not file_path:
            return
        
        data = {
            "client_id": self.client_id,
            "documents": {
                doc_id: {
                    "content": doc.content,
                    "tokens": doc.tokens,
                    "metadata": doc.metadata,
                }
                for doc_id, doc in self.documents.items()
            },
            "doc_order": self.doc_order,
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        logger.debug(f"Persisted BM25 index to {file_path}")
    
    def _load_from_disk(self) -> None:
        """Load index from disk."""
        file_path = self._get_persist_file()
        if not file_path or not file_path.exists():
            return
        
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            self.client_id = data.get("client_id")
            self.doc_order = data.get("doc_order", [])
            
            for doc_id, doc_data in data.get("documents", {}).items():
                self.documents[doc_id] = BM25Document(
                    id=doc_id,
                    content=doc_data["content"],
                    tokens=doc_data["tokens"],
                    metadata=doc_data.get("metadata", {}),
                )
            
            self._dirty = True  # Need to rebuild BM25 index
            logger.info(f"Loaded BM25 index from {file_path} with {len(self.documents)} docs")
            
        except Exception as e:
            logger.error(f"Failed to load BM25 index from {file_path}: {e}")


# Global index cache
_bm25_indices: Dict[str, BM25Index] = {}


def get_bm25_index(
    client_id: Optional[str] = None,
    persist_path: str = "./data/bm25",
) -> BM25Index:
    """
    Get or create a BM25 index.
    
    Args:
        client_id: Client ID for per-client index
        persist_path: Directory for persistence
        
    Returns:
        BM25Index instance
    """
    cache_key = client_id or "_global_"
    
    if cache_key not in _bm25_indices:
        _bm25_indices[cache_key] = BM25Index(
            persist_path=persist_path,
            client_id=client_id,
        )
    
    return _bm25_indices[cache_key]


def clear_bm25_cache() -> None:
    """Clear the BM25 index cache."""
    global _bm25_indices
    _bm25_indices = {}
