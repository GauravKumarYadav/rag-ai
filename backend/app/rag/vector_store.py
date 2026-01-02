from functools import lru_cache
from typing import Dict, List, Optional
import re

import chromadb

from app.config import settings
from app.models.schemas import RetrievalHit
from app.rag.embeddings import get_embedding_function


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize a string to be a valid ChromaDB collection name.
    Rules: 3-63 chars, alphanumeric + underscores + hyphens, start/end with alphanum.
    """
    # Replace spaces and special chars with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_-')
    # Ensure starts with letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'c_' + sanitized
    # Ensure min length
    if len(sanitized) < 3:
        sanitized = sanitized + '_col'
    # Truncate if too long
    if len(sanitized) > 63:
        sanitized = sanitized[:63].rstrip('_-')
    return sanitized


class VectorStore:
    """Base vector store for global documents and memories."""
    
    def __init__(self, path: str) -> None:
        self.path = path
        self.client = chromadb.PersistentClient(path=path)
        self.embedding_fn = get_embedding_function()
        self.docs = self.client.get_or_create_collection(
            name="documents", embedding_function=self.embedding_fn
        )
        self.memories = self.client.get_or_create_collection(
            name="memories", embedding_function=self.embedding_fn
        )

    def add_documents(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        self.docs.add(documents=contents, ids=ids, metadatas=metadatas)

    def add_memories(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        self.memories.add(documents=contents, ids=ids, metadatas=metadatas)

    def query(
        self,
        query: str,
        top_k: int = 4,
        where: Optional[Dict] = None,
        collection: str = "documents",
    ) -> List[RetrievalHit]:
        target = self.docs if collection == "documents" else self.memories
        results = target.query(query_texts=[query], n_results=top_k, where=where)
        hits: List[RetrievalHit] = []
        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for doc_id, content, meta, score in zip(ids, docs, metadatas, distances):
            hits.append(RetrievalHit(id=doc_id, content=content, score=score, metadata=meta or {}))
        return hits


class ClientVectorStore:
    """
    Per-client vector store with isolated collections.
    Each client gets their own documents and memories collections.
    """
    
    def __init__(self, path: str, client_id: str) -> None:
        self.path = path
        self.client_id = client_id
        self.chroma_client = chromadb.PersistentClient(path=path)
        self.embedding_fn = get_embedding_function()
        
        # Create client-specific collection names
        safe_id = sanitize_collection_name(client_id)
        self.docs_name = f"client_{safe_id}_docs"
        self.memories_name = f"client_{safe_id}_memories"
        
        self.docs = self.chroma_client.get_or_create_collection(
            name=self.docs_name, embedding_function=self.embedding_fn
        )
        self.memories = self.chroma_client.get_or_create_collection(
            name=self.memories_name, embedding_function=self.embedding_fn
        )
    
    def add_documents(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        """Add documents to client's collection."""
        # Ensure client_id is in metadata
        if metadatas:
            for meta in metadatas:
                meta["client_id"] = self.client_id
        else:
            metadatas = [{"client_id": self.client_id} for _ in contents]
        
        self.docs.add(documents=contents, ids=ids, metadatas=metadatas)
    
    def add_memories(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        """Add memories to client's collection."""
        if metadatas:
            for meta in metadatas:
                meta["client_id"] = self.client_id
        else:
            metadatas = [{"client_id": self.client_id} for _ in contents]
        
        self.memories.add(documents=contents, ids=ids, metadatas=metadatas)
    
    def query(
        self,
        query: str,
        top_k: int = 4,
        where: Optional[Dict] = None,
        collection: str = "documents",
    ) -> List[RetrievalHit]:
        """Query client's collection."""
        target = self.docs if collection == "documents" else self.memories
        results = target.query(query_texts=[query], n_results=top_k, where=where)
        hits: List[RetrievalHit] = []
        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for doc_id, content, meta, score in zip(ids, docs, metadatas, distances):
            hits.append(RetrievalHit(id=doc_id, content=content, score=score, metadata=meta or {}))
        return hits
    
    def get_stats(self) -> Dict:
        """Get statistics for this client's collections."""
        return {
            "client_id": self.client_id,
            "document_count": self.docs.count(),
            "memory_count": self.memories.count(),
            "docs_collection": self.docs_name,
            "memories_collection": self.memories_name,
        }
    
    def delete_all(self) -> None:
        """Delete all documents and memories for this client."""
        # Delete collections entirely
        try:
            self.chroma_client.delete_collection(self.docs_name)
        except Exception:
            pass
        try:
            self.chroma_client.delete_collection(self.memories_name)
        except Exception:
            pass


# Cache for client vector stores
_client_stores: Dict[str, ClientVectorStore] = {}


def get_client_vector_store(client_id: str) -> ClientVectorStore:
    """Get or create a vector store for a specific client."""
    if client_id not in _client_stores:
        _client_stores[client_id] = ClientVectorStore(
            path=settings.chroma_db_path,
            client_id=client_id,
        )
    return _client_stores[client_id]


def clear_client_vector_store_cache(client_id: Optional[str] = None) -> None:
    """Clear cached client vector stores."""
    global _client_stores
    if client_id:
        _client_stores.pop(client_id, None)
    else:
        _client_stores.clear()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Get the global vector store (for backward compatibility)."""
    return VectorStore(path=settings.chroma_db_path)

