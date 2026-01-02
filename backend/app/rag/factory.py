"""
Vector Store Factory - Provider selection and instantiation.

This module provides a factory pattern for creating vector store instances
based on configuration. To add a new provider:

1. Create implementation in app/rag/ (e.g., pinecone_store.py)
2. Import and register in PROVIDERS dict below
3. Set VECTOR_STORE_PROVIDER in config/env

Supported providers:
- chromadb: Local embedded database (default)
- pinecone: Cloud vector database (TODO)
- weaviate: Self-hosted or cloud (TODO)
- qdrant: High-performance vector search (TODO)
- milvus: Scalable vector database (TODO)
"""

from functools import lru_cache
from typing import Dict, Optional, Type

from app.config import settings
from app.rag.base import VectorStoreBase, ClientVectorStoreBase, VectorStoreConfig
from app.rag.chroma_store import ChromaVectorStore, ChromaClientVectorStore


# Registry of available providers
# Add new providers here after implementing them
PROVIDERS: Dict[str, Type[VectorStoreBase]] = {
    "chromadb": ChromaVectorStore,
    "chroma": ChromaVectorStore,  # Alias
}

CLIENT_PROVIDERS: Dict[str, Type[ClientVectorStoreBase]] = {
    "chromadb": ChromaClientVectorStore,
    "chroma": ChromaClientVectorStore,
}


def get_vector_store_config() -> VectorStoreConfig:
    """Build configuration from settings."""
    return VectorStoreConfig(
        path=settings.chroma_db_path,
        url=getattr(settings, 'vector_store_url', None),
        api_key=getattr(settings, 'vector_store_api_key', None),
        namespace=getattr(settings, 'vector_store_namespace', None),
        collection_prefix=getattr(settings, 'vector_store_prefix', ''),
    )


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreBase:
    """
    Get the global vector store instance.
    
    Uses the provider specified in settings.vector_store_provider.
    Defaults to 'chromadb' if not specified.
    """
    provider = getattr(settings, 'vector_store_provider', 'chromadb').lower()
    
    if provider not in PROVIDERS:
        available = ', '.join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown vector store provider: '{provider}'. "
            f"Available providers: {available}"
        )
    
    store_class = PROVIDERS[provider]
    config = get_vector_store_config()
    
    return store_class(config)


# Cache for client-specific stores
_client_stores: Dict[str, ClientVectorStoreBase] = {}


def get_client_vector_store(client_id: str) -> ClientVectorStoreBase:
    """
    Get or create a vector store for a specific client.
    
    Each client gets isolated collections for data separation.
    """
    if client_id not in _client_stores:
        provider = getattr(settings, 'vector_store_provider', 'chromadb').lower()
        
        if provider not in CLIENT_PROVIDERS:
            available = ', '.join(CLIENT_PROVIDERS.keys())
            raise ValueError(
                f"Unknown vector store provider: '{provider}'. "
                f"Available providers: {available}"
            )
        
        store_class = CLIENT_PROVIDERS[provider]
        config = get_vector_store_config()
        
        _client_stores[client_id] = store_class(config, client_id)
    
    return _client_stores[client_id]


def clear_client_vector_store_cache(client_id: Optional[str] = None) -> None:
    """
    Clear cached client vector stores.
    
    Args:
        client_id: Specific client to clear, or None to clear all
    """
    global _client_stores
    if client_id:
        _client_stores.pop(client_id, None)
    else:
        _client_stores.clear()


def list_providers() -> Dict[str, str]:
    """List available vector store providers with their status."""
    return {
        "chromadb": "available",
        "pinecone": "not_implemented",
        "weaviate": "not_implemented", 
        "qdrant": "not_implemented",
        "milvus": "not_implemented",
    }


# =============================================================================
# Example: Adding a new provider (Pinecone)
# =============================================================================
#
# 1. Create app/rag/pinecone_store.py:
#
#    from app.rag.base import VectorStoreBase, VectorStoreConfig
#    import pinecone
#
#    class PineconeVectorStore(VectorStoreBase):
#        def __init__(self, config: VectorStoreConfig):
#            pinecone.init(api_key=config.api_key)
#            self.index = pinecone.Index(config.namespace or "default")
#            ...
#
# 2. Register in this file:
#
#    from app.rag.pinecone_store import PineconeVectorStore
#    PROVIDERS["pinecone"] = PineconeVectorStore
#
# 3. Configure in .env:
#
#    VECTOR_STORE_PROVIDER=pinecone
#    VECTOR_STORE_API_KEY=your-pinecone-key
#    VECTOR_STORE_NAMESPACE=your-index-name
#
