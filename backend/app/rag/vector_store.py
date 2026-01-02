"""
Vector Store Module - Unified access to vector database.

This module provides backward-compatible access to vector stores while
supporting the new modular architecture. All imports from this module
will work as before, but now use the factory pattern internally.

For direct access to the modular system, use:
    from app.rag.factory import get_vector_store, get_client_vector_store
    from app.rag.base import VectorStoreBase, VectorStoreConfig
"""

# Re-export from factory for backward compatibility
from app.rag.factory import (
    get_vector_store,
    get_client_vector_store,
    clear_client_vector_store_cache,
    list_providers,
    get_vector_store_config,
)

# Re-export base classes for type hints
from app.rag.base import (
    VectorStoreBase,
    ClientVectorStoreBase,
    VectorStoreConfig,
)

# Re-export ChromaDB implementation for direct use if needed
from app.rag.chroma_store import (
    ChromaVectorStore,
    ChromaClientVectorStore,
    sanitize_collection_name,
)

# Legacy aliases for complete backward compatibility
VectorStore = ChromaVectorStore
ClientVectorStore = ChromaClientVectorStore

__all__ = [
    # Factory functions
    "get_vector_store",
    "get_client_vector_store", 
    "clear_client_vector_store_cache",
    "list_providers",
    "get_vector_store_config",
    
    # Base classes
    "VectorStoreBase",
    "ClientVectorStoreBase",
    "VectorStoreConfig",
    
    # ChromaDB implementation
    "ChromaVectorStore",
    "ChromaClientVectorStore",
    "sanitize_collection_name",
    
    # Legacy aliases
    "VectorStore",
    "ClientVectorStore",
]
