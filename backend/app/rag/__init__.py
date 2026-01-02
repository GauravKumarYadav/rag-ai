# Retrieval package - Modular vector store architecture
#
# Usage:
#   from app.rag import get_vector_store, get_client_vector_store
#   from app.rag import VectorStoreBase  # For type hints
#
# To switch providers, set VECTOR_STORE_PROVIDER in .env:
#   VECTOR_STORE_PROVIDER=chromadb  (default)
#   VECTOR_STORE_PROVIDER=pinecone
#   VECTOR_STORE_PROVIDER=weaviate

from app.rag.vector_store import (
    get_vector_store,
    get_client_vector_store,
    clear_client_vector_store_cache,
    list_providers,
    VectorStoreBase,
    ClientVectorStoreBase,
    VectorStoreConfig,
)

from app.rag.retriever import get_retriever, Retriever
from app.rag.embeddings import get_embedding_function

__all__ = [
    "get_vector_store",
    "get_client_vector_store",
    "clear_client_vector_store_cache",
    "list_providers",
    "VectorStoreBase",
    "ClientVectorStoreBase",
    "VectorStoreConfig",
    "get_retriever",
    "Retriever",
    "get_embedding_function",
]
