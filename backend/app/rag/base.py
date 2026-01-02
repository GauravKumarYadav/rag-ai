"""
Abstract base classes for vector store implementations.

This module defines the interface that all vector store implementations must follow.
To add a new vector database:
1. Create a new file in app/rag/ (e.g., pinecone_store.py)
2. Implement the VectorStoreBase interface
3. Register it in the factory (app/rag/factory.py)
4. Set VECTOR_STORE_PROVIDER in config
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.models.schemas import RetrievalHit


@dataclass
class VectorStoreConfig:
    """Configuration for vector store initialization."""
    path: Optional[str] = None  # For file-based stores like ChromaDB
    url: Optional[str] = None   # For cloud-based stores
    api_key: Optional[str] = None
    namespace: Optional[str] = None
    collection_prefix: str = ""
    extra: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class VectorStoreBase(ABC):
    """
    Abstract base class for vector store implementations.
    
    All vector databases must implement this interface to be compatible
    with the RAG pipeline. This allows swapping ChromaDB for Pinecone,
    Weaviate, Qdrant, Milvus, etc. without changing application code.
    """
    
    @abstractmethod
    def __init__(self, config: VectorStoreConfig) -> None:
        """Initialize the vector store with configuration."""
        pass
    
    @abstractmethod
    def add_documents(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """
        Add documents to the store.
        
        Args:
            contents: List of document texts
            ids: Unique IDs for each document
            metadatas: Optional metadata dicts for each document
            embeddings: Pre-computed embeddings (if None, store computes them)
        """
        pass
    
    @abstractmethod
    def add_memories(
        self,
        contents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """
        Add memories (conversation summaries) to the store.
        
        Args:
            contents: List of memory texts
            ids: Unique IDs for each memory
            metadatas: Optional metadata dicts for each memory
            embeddings: Pre-computed embeddings (if None, store computes them)
        """
        pass
    
    @abstractmethod
    def query(
        self,
        query: str,
        top_k: int = 4,
        where: Optional[Dict] = None,
        collection: str = "documents",
    ) -> List[RetrievalHit]:
        """
        Query the vector store for similar documents.
        
        Args:
            query: The search query text
            top_k: Number of results to return
            where: Optional metadata filters
            collection: Which collection to query ("documents" or "memories")
            
        Returns:
            List of RetrievalHit objects sorted by relevance
        """
        pass
    
    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        collection: str = "documents",
    ) -> int:
        """
        Delete documents from the store.
        
        Args:
            ids: Specific document IDs to delete
            where: Metadata filter for deletion
            collection: Which collection to delete from
            
        Returns:
            Number of documents deleted
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict with keys like 'document_count', 'memory_count', etc.
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the vector store is healthy and connected.
        
        Returns:
            True if healthy, False otherwise
        """
        pass


class ClientVectorStoreBase(VectorStoreBase):
    """
    Extended interface for per-client isolated vector stores.
    
    Implementations should ensure complete data isolation between clients.
    """
    
    def __init__(self, config: VectorStoreConfig, client_id: str) -> None:
        """Initialize with client-specific isolation."""
        self.client_id = client_id
        super().__init__(config)
    
    @abstractmethod
    def delete_all(self) -> None:
        """Delete all data for this client (GDPR compliance)."""
        pass
    
    @abstractmethod
    def export_data(self) -> Dict[str, Any]:
        """Export all client data (GDPR data portability)."""
        pass
