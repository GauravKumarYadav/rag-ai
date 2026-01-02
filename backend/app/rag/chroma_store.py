"""
ChromaDB implementation of the vector store interface.

ChromaDB is a lightweight, embedded vector database ideal for:
- Local development
- Single-machine deployments
- Privacy-focused applications (all data stays local)

For production with high availability, consider Pinecone, Weaviate, or Qdrant.
"""

import re
from typing import Any, Dict, List, Optional

import chromadb

from app.rag.base import VectorStoreBase, ClientVectorStoreBase, VectorStoreConfig
from app.rag.embeddings import get_embedding_function
from app.models.schemas import RetrievalHit


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
    """
    
    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self.path = config.path
        
        # Initialize ChromaDB client
        if config.url:
            # Remote ChromaDB server
            self.client = chromadb.HttpClient(host=config.url)
        else:
            # Local persistent storage
            self.client = chromadb.PersistentClient(path=self.path)
        
        self.embedding_fn = get_embedding_function()
        
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


class ChromaClientVectorStore(ClientVectorStoreBase):
    """
    Per-client isolated ChromaDB vector store.
    
    Each client gets separate collections for complete data isolation.
    """
    
    def __init__(self, config: VectorStoreConfig, client_id: str) -> None:
        self.client_id = client_id
        self.config = config
        self.path = config.path
        
        # Initialize ChromaDB client
        if config.url:
            self.chroma_client = chromadb.HttpClient(host=config.url)
        else:
            self.chroma_client = chromadb.PersistentClient(path=self.path)
        
        self.embedding_fn = get_embedding_function()
        
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
