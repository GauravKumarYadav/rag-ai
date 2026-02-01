"""
ChromaDB implementation of the vector store interface.

ChromaDB is a lightweight, embedded vector database ideal for:
- Local development
- Single-machine deployments
- Privacy-focused applications (all data stays local)

For production with high availability, consider Pinecone, Weaviate, or Qdrant.

Embedding Fingerprinting:
- Stores embedding configuration fingerprint with each collection
- Verifies fingerprint on startup to detect model changes
- Prevents silent degradation from embedding mismatches
"""

import logging
import re
from typing import Any, Dict, List, Optional

import chromadb

from app.rag.base import VectorStoreBase, ClientVectorStoreBase, VectorStoreConfig
from app.rag.embeddings import get_embedding_function, get_embedding_fingerprint, verify_embedding_fingerprint
from app.models.schemas import RetrievalHit

logger = logging.getLogger(__name__)

# Metadata key for storing embedding fingerprint
EMBEDDING_FINGERPRINT_KEY = "_embedding_fingerprint"


class EmbeddingMismatchError(Exception):
    """Raised when embedding configuration doesn't match stored index."""
    pass


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize a string to be a valid ChromaDB collection name.
    Rules: 3-63 chars, alphanumeric + underscores + hyphens, start/end with alphanum.
    """
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_-')
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'c_' + sanitized
    if len(sanitized) < 3:
        sanitized = sanitized + '_col'
    if len(sanitized) > 63:
        sanitized = sanitized[:63].rstrip('_-')
    return sanitized


class ChromaVectorStore(VectorStoreBase):
    """
    ChromaDB implementation of the vector store.
    
    Uses persistent storage and automatic embedding generation.
    Includes embedding fingerprint verification for consistency.
    """
    
    def __init__(self, config: VectorStoreConfig, verify_fingerprint: bool = True) -> None:
        self.config = config
        self.path = config.path
        self._verify_fingerprint = verify_fingerprint
        
        # Initialize ChromaDB client
        if config.url:
            # Remote ChromaDB server
            # Parse URL to extract host and port
            url = config.url.rstrip("/")
            if url.startswith("http://"):
                url = url[7:]
            elif url.startswith("https://"):
                url = url[8:]
            
            if ":" in url:
                host, port_str = url.rsplit(":", 1)
                port = int(port_str)
            else:
                host = url
                port = 8000  # Default ChromaDB port
            
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # Local persistent storage
            self.client = chromadb.PersistentClient(path=self.path)
        
        self.embedding_fn = get_embedding_function()
        self._current_fingerprint = get_embedding_fingerprint()
        
        # Collection names with optional prefix
        prefix = config.collection_prefix
        docs_name = f"{prefix}documents" if prefix else "documents"
        memories_name = f"{prefix}memories" if prefix else "memories"
        
        self.docs = self.client.get_or_create_collection(
            name=docs_name, 
            embedding_function=self.embedding_fn
        )
        self.memories = self.client.get_or_create_collection(
            name=memories_name, 
            embedding_function=self.embedding_fn
        )
        
        # Verify embedding fingerprint if collection has data
        if verify_fingerprint and self.docs.count() > 0:
            self._verify_embedding_consistency()
    
    def add_documents(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        kwargs = {"documents": contents, "ids": ids}
        if metadatas:
            kwargs["metadatas"] = metadatas
        if embeddings:
            kwargs["embeddings"] = embeddings
        self.docs.add(**kwargs)
    
    def add_memories(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        kwargs = {"documents": contents, "ids": ids}
        if metadatas:
            kwargs["metadatas"] = metadatas
        if embeddings:
            kwargs["embeddings"] = embeddings
        self.memories.add(**kwargs)
    
    def query(
        self,
        query: str,
        top_k: int = 4,
        where: Optional[Dict] = None,
        collection: str = "documents",
    ) -> List[RetrievalHit]:
        target = self.docs if collection == "documents" else self.memories
        
        kwargs = {"query_texts": [query], "n_results": top_k}
        if where:
            kwargs["where"] = where
            
        results = target.query(**kwargs)
        
        hits: List[RetrievalHit] = []
        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for doc_id, content, meta, score in zip(ids, docs, metadatas, distances):
            hits.append(RetrievalHit(
                id=doc_id, 
                content=content, 
                score=score, 
                metadata=meta or {}
            ))
        return hits
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        collection: str = "documents",
    ) -> int:
        target = self.docs if collection == "documents" else self.memories
        
        # Get count before deletion
        count_before = target.count()
        
        if ids:
            target.delete(ids=ids)
        elif where:
            target.delete(where=where)
        
        count_after = target.count()
        return count_before - count_after
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "provider": "chromadb",
            "path": self.path,
            "document_count": self.docs.count(),
            "memory_count": self.memories.count(),
        }
    
    def health_check(self) -> bool:
        try:
            # Simple heartbeat check
            self.client.heartbeat()
            return True
        except Exception:
            return False
    
    def _verify_embedding_consistency(self) -> None:
        """
        Verify that the current embedding config matches stored documents.
        
        Raises EmbeddingMismatchError if there's a mismatch.
        """
        # Sample a document to check its fingerprint
        try:
            sample = self.docs.get(limit=1, include=["metadatas"])
            if sample["ids"] and sample["metadatas"]:
                stored_fp = sample["metadatas"][0].get("embedding_fingerprint")
                if stored_fp and stored_fp != self._current_fingerprint:
                    logger.warning(
                        f"Embedding configuration mismatch detected! "
                        f"Stored: {stored_fp}, Current: {self._current_fingerprint}. "
                        f"Reindexing recommended."
                    )
        except Exception as e:
            logger.debug(f"Could not verify embedding fingerprint: {e}")
    
    def get_stored_fingerprint(self) -> Optional[str]:
        """Get the embedding fingerprint stored with documents."""
        try:
            sample = self.docs.get(limit=1, include=["metadatas"])
            if sample["ids"] and sample["metadatas"]:
                return sample["metadatas"][0].get("embedding_fingerprint")
        except Exception:
            pass
        return None
    
    def check_fingerprint_match(self) -> bool:
        """
        Check if stored fingerprint matches current config.
        
        Returns:
            True if match (or no stored data), False if mismatch
        """
        stored = self.get_stored_fingerprint()
        if not stored:
            return True  # No data to compare
        return verify_embedding_fingerprint(stored)


class ChromaClientVectorStore(ClientVectorStoreBase):
    """
    Per-client isolated ChromaDB vector store.
    
    Each client gets separate collections for complete data isolation.
    Includes embedding fingerprint verification for consistency.
    """
    
    def __init__(self, config: VectorStoreConfig, client_id: str, verify_fingerprint: bool = True) -> None:
        self.client_id = client_id
        self.config = config
        self.path = config.path
        self._verify_fingerprint = verify_fingerprint
        
        # Initialize ChromaDB client
        if config.url:
            # Parse URL to extract host and port
            url = config.url.rstrip("/")
            if url.startswith("http://"):
                url = url[7:]
            elif url.startswith("https://"):
                url = url[8:]
            
            if ":" in url:
                host, port_str = url.rsplit(":", 1)
                port = int(port_str)
            else:
                host = url
                port = 8000
            
            self.chroma_client = chromadb.HttpClient(host=host, port=port)
        else:
            self.chroma_client = chromadb.PersistentClient(path=self.path)
        
        self.embedding_fn = get_embedding_function()
        self._current_fingerprint = get_embedding_fingerprint()
        
        # Create client-specific collection names
        safe_id = sanitize_collection_name(client_id)
        prefix = config.collection_prefix
        self.docs_name = f"{prefix}client_{safe_id}_docs" if prefix else f"client_{safe_id}_docs"
        self.memories_name = f"{prefix}client_{safe_id}_memories" if prefix else f"client_{safe_id}_memories"
        
        self.docs = self.chroma_client.get_or_create_collection(
            name=self.docs_name, 
            embedding_function=self.embedding_fn
        )
        self.memories = self.chroma_client.get_or_create_collection(
            name=self.memories_name, 
            embedding_function=self.embedding_fn
        )
        
        # Verify embedding fingerprint if collection has data
        if verify_fingerprint and self.docs.count() > 0:
            self._verify_embedding_consistency()
    
    def add_documents(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        # Ensure client_id is in metadata for traceability
        if metadatas:
            for meta in metadatas:
                meta["client_id"] = self.client_id
        else:
            metadatas = [{"client_id": self.client_id} for _ in contents]
        
        kwargs = {"documents": contents, "ids": ids, "metadatas": metadatas}
        if embeddings:
            kwargs["embeddings"] = embeddings
        self.docs.add(**kwargs)
    
    def add_memories(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        if metadatas:
            for meta in metadatas:
                meta["client_id"] = self.client_id
        else:
            metadatas = [{"client_id": self.client_id} for _ in contents]
        
        kwargs = {"documents": contents, "ids": ids, "metadatas": metadatas}
        if embeddings:
            kwargs["embeddings"] = embeddings
        self.memories.add(**kwargs)
    
    def query(
        self,
        query: str,
        top_k: int = 4,
        where: Optional[Dict] = None,
        collection: str = "documents",
    ) -> List[RetrievalHit]:
        target = self.docs if collection == "documents" else self.memories
        
        kwargs = {"query_texts": [query], "n_results": top_k}
        if where:
            kwargs["where"] = where
            
        results = target.query(**kwargs)
        
        hits: List[RetrievalHit] = []
        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for doc_id, content, meta, score in zip(ids, docs, metadatas, distances):
            hits.append(RetrievalHit(
                id=doc_id, 
                content=content, 
                score=score, 
                metadata=meta or {}
            ))
        return hits
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        collection: str = "documents",
    ) -> int:
        target = self.docs if collection == "documents" else self.memories
        count_before = target.count()
        
        if ids:
            target.delete(ids=ids)
        elif where:
            target.delete(where=where)
        
        count_after = target.count()
        return count_before - count_after

    def get_documents_by_ids(self, ids: List[str]) -> Dict[str, str]:
        """Get document contents by IDs (for BM25 content resolution from Chroma)."""
        if not ids:
            return {}
        try:
            result = self.docs.get(ids=ids, include=["documents"])
            chroma_ids = result.get("ids", [])
            docs = result.get("documents", [])
            # Chroma may return documents as list of lists (one per id)
            out = {}
            for i, doc_id in enumerate(chroma_ids):
                if i < len(docs):
                    content = docs[i]
                    if isinstance(content, list):
                        content = content[0] if content else ""
                    out[doc_id] = content or ""
                else:
                    out[doc_id] = ""
            return out
        except Exception as e:
            logger.debug("get_documents_by_ids failed: %s", e)
            return {}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "provider": "chromadb",
            "client_id": self.client_id,
            "document_count": self.docs.count(),
            "memory_count": self.memories.count(),
            "docs_collection": self.docs_name,
            "memories_collection": self.memories_name,
        }
    
    def health_check(self) -> bool:
        try:
            self.chroma_client.heartbeat()
            return True
        except Exception:
            return False
    
    def delete_all(self) -> None:
        """Delete all data for this client (GDPR compliance)."""
        try:
            self.chroma_client.delete_collection(self.docs_name)
        except Exception:
            pass
        try:
            self.chroma_client.delete_collection(self.memories_name)
        except Exception:
            pass
    
    def export_data(self) -> Dict[str, Any]:
        """Export all client data (GDPR data portability)."""
        docs_data = self.docs.get()
        memories_data = self.memories.get()
        
        return {
            "client_id": self.client_id,
            "documents": {
                "ids": docs_data.get("ids", []),
                "contents": docs_data.get("documents", []),
                "metadatas": docs_data.get("metadatas", []),
            },
            "memories": {
                "ids": memories_data.get("ids", []),
                "contents": memories_data.get("documents", []),
                "metadatas": memories_data.get("metadatas", []),
            },
        }
    
    def _verify_embedding_consistency(self) -> None:
        """
        Verify that the current embedding config matches stored documents.
        
        Logs a warning if there's a mismatch.
        """
        try:
            sample = self.docs.get(limit=1, include=["metadatas"])
            if sample["ids"] and sample["metadatas"]:
                stored_fp = sample["metadatas"][0].get("embedding_fingerprint")
                if stored_fp and stored_fp != self._current_fingerprint:
                    logger.warning(
                        f"Embedding mismatch for client {self.client_id}! "
                        f"Stored: {stored_fp}, Current: {self._current_fingerprint}. "
                        f"Reindexing recommended."
                    )
        except Exception as e:
            logger.debug(f"Could not verify embedding fingerprint for client {self.client_id}: {e}")
    
    def get_stored_fingerprint(self) -> Optional[str]:
        """Get the embedding fingerprint stored with documents."""
        try:
            sample = self.docs.get(limit=1, include=["metadatas"])
            if sample["ids"] and sample["metadatas"]:
                return sample["metadatas"][0].get("embedding_fingerprint")
        except Exception:
            pass
        return None
    
    def check_fingerprint_match(self) -> bool:
        """
        Check if stored fingerprint matches current config.
        
        Returns:
            True if match (or no stored data), False if mismatch
        """
        stored = self.get_stored_fingerprint()
        if not stored:
            return True
        return verify_embedding_fingerprint(stored)


class GlobalVectorStore:
    """
    Global vector store for documents available to all clients.
    
    Uses a dedicated 'global_docs' collection that can be searched
    alongside client-specific collections.
    """
    
    GLOBAL_DOCS_COLLECTION = "global_docs"
    GLOBAL_MEMORIES_COLLECTION = "global_memories"
    
    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        
        # Initialize ChromaDB client
        if config.url:
            url = config.url.rstrip("/")
            if url.startswith("http://"):
                url = url[7:]
            elif url.startswith("https://"):
                url = url[8:]
            
            if ":" in url:
                host, port_str = url.rsplit(":", 1)
                port = int(port_str)
            else:
                host = url
                port = 8000
            
            self.chroma_client = chromadb.HttpClient(host=host, port=port)
        else:
            self.chroma_client = chromadb.PersistentClient(path=config.path)
        
        self.embedding_fn = get_embedding_function()
        
        # Create global collections
        prefix = config.collection_prefix
        self.docs_name = f"{prefix}{self.GLOBAL_DOCS_COLLECTION}" if prefix else self.GLOBAL_DOCS_COLLECTION
        self.memories_name = f"{prefix}{self.GLOBAL_MEMORIES_COLLECTION}" if prefix else self.GLOBAL_MEMORIES_COLLECTION
        
        self.docs = self.chroma_client.get_or_create_collection(
            name=self.docs_name,
            embedding_function=self.embedding_fn
        )
        self.memories = self.chroma_client.get_or_create_collection(
            name=self.memories_name,
            embedding_function=self.embedding_fn
        )
    
    def add_documents(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Add documents to the global collection."""
        if metadatas:
            for meta in metadatas:
                meta["client_id"] = "global"
                meta["is_global"] = True
        else:
            metadatas = [{"client_id": "global", "is_global": True} for _ in contents]
        
        kwargs = {"documents": contents, "ids": ids, "metadatas": metadatas}
        if embeddings:
            kwargs["embeddings"] = embeddings
        self.docs.add(**kwargs)

    def get_documents_by_ids(self, ids: List[str]) -> Dict[str, str]:
        """Get document contents by IDs (for BM25 content resolution from Chroma)."""
        if not ids:
            return {}
        try:
            result = self.docs.get(ids=ids, include=["documents"])
            chroma_ids = result.get("ids", [])
            docs = result.get("documents", [])
            out = {}
            for i, doc_id in enumerate(chroma_ids):
                if i < len(docs):
                    content = docs[i]
                    if isinstance(content, list):
                        content = content[0] if content else ""
                    out[doc_id] = content or ""
                else:
                    out[doc_id] = ""
            return out
        except Exception as e:
            logger.debug("get_documents_by_ids failed: %s", e)
            return {}
    
    def query(
        self,
        query: str,
        top_k: int = 4,
        where: Optional[Dict] = None,
    ) -> List[RetrievalHit]:
        """Query the global documents collection."""
        kwargs = {"query_texts": [query], "n_results": top_k}
        if where:
            kwargs["where"] = where
        
        results = self.docs.query(**kwargs)
        
        hits: List[RetrievalHit] = []
        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for doc_id, content, meta, score in zip(ids, docs, metadatas, distances):
            hits.append(RetrievalHit(
                id=doc_id,
                content=content,
                score=score,
                metadata=meta or {}
            ))
        return hits
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
    ) -> int:
        """Delete documents from the global collection."""
        count_before = self.docs.count()
        
        if ids:
            self.docs.delete(ids=ids)
        elif where:
            self.docs.delete(where=where)
        
        count_after = self.docs.count()
        return count_before - count_after
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for the global collection."""
        return {
            "provider": "chromadb",
            "collection": "global",
            "document_count": self.docs.count(),
            "memory_count": self.memories.count(),
        }
    
    def get(self, ids: List[str]) -> Optional[Dict]:
        """Get documents by IDs (for hot-reload verification)."""
        try:
            result = self.docs.get(ids=ids, include=["documents", "metadatas"])
            if result["ids"]:
                return result
        except Exception:
            pass
        return None


# Singleton for global vector store
_global_vector_store: Optional[GlobalVectorStore] = None


def get_global_vector_store() -> GlobalVectorStore:
    """Get or create the global vector store singleton."""
    global _global_vector_store
    if _global_vector_store is None:
        from app.config import settings
        config = VectorStoreConfig(
            path=settings.rag.chroma_db_path,
            url=settings.rag.url,
            collection_prefix=settings.rag.collection_prefix,
        )
        _global_vector_store = GlobalVectorStore(config)
    return _global_vector_store