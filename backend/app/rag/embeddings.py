"""
Embedding functions for vector stores.

Supports multiple embedding providers:
- LMStudio (OpenAI-compatible API)
- Ollama (native embeddings API)
- Embedding Service (microservices mode)

Includes embedding fingerprinting for consistency verification.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import httpx
import numpy as np
from chromadb import EmbeddingFunction, Documents, Embeddings

from app.config import settings


# =============================================================================
# Embedding Fingerprint
# =============================================================================

@dataclass
class EmbeddingFingerprint:
    """
    Fingerprint for tracking embedding model configuration.
    
    Used to detect when the embedding model changes, which would
    require re-indexing documents for consistency.
    """
    provider: str          # "ollama", "lmstudio", "embedding-service"
    model: str             # "nomic-embed-text"
    dimension: int         # 768
    normalize: bool        # True
    version: str = "1.0"   # Configuration version
    
    def to_string(self) -> str:
        """Get fingerprint as a string for storage."""
        return f"{self.provider}:{self.model}:{self.dimension}:{self.normalize}:{self.version}"
    
    @classmethod
    def from_string(cls, s: str) -> "EmbeddingFingerprint":
        """Parse fingerprint from stored string."""
        parts = s.split(":")
        if len(parts) < 4:
            raise ValueError(f"Invalid fingerprint format: {s}")
        
        return cls(
            provider=parts[0],
            model=parts[1],
            dimension=int(parts[2]),
            normalize=parts[3].lower() == "true",
            version=parts[4] if len(parts) > 4 else "1.0",
        )
    
    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.to_string() == other
        if isinstance(other, EmbeddingFingerprint):
            return self.to_string() == other.to_string()
        return False
    
    def __hash__(self) -> int:
        return hash(self.to_string())


def get_embedding_fingerprint() -> str:
    """
    Get the current embedding configuration fingerprint.
    
    This is stored with each chunk to detect when re-indexing is needed.
    
    Returns:
        Fingerprint string like "ollama:nomic-embed-text:768:true:1.0"
    """
    fp = EmbeddingFingerprint(
        provider=settings.rag.embedding_provider,
        model=settings.rag.embedding_model,
        dimension=settings.rag.embedding_dimension,
        normalize=settings.rag.embedding_normalize,
    )
    return fp.to_string()


def verify_embedding_fingerprint(stored_fingerprint: str) -> bool:
    """
    Verify that the stored fingerprint matches current configuration.
    
    Args:
        stored_fingerprint: The fingerprint stored with indexed documents
        
    Returns:
        True if fingerprints match, False if re-indexing is needed
    """
    current = get_embedding_fingerprint()
    return current == stored_fingerprint


class LMStudioEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function using LMStudio's local embedding API (OpenAI-compatible)."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "text-embedding-nomic-embed-text-v1.5",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=60.0)
    
    def name(self) -> str:
        """Return the name of the embedding function for ChromaDB."""
        return "lmstudio"
    
    def __call__(self, input: Documents) -> Embeddings:
        """ChromaDB calls embedding functions with __call__."""
        return self.embed_documents(input)
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of documents."""
        if not texts:
            return []
        
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = self._client.post(
                f"{self.base_url}/embeddings",
                json={"input": texts, "model": self.model},
            )
            response.raise_for_status()
            data = response.json()
            
            # Sort by index to ensure correct order
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [np.array(item["embedding"], dtype=np.float32) for item in embeddings]
        except Exception as e:
            print(f"Embedding error for texts: {texts[:100] if texts else texts}, error: {e}")
            raise
    
    def embed_query(self, input) -> List[np.ndarray]:
        """Embed query texts. ChromaDB expects List[np.ndarray] for queries."""
        # ChromaDB passes a list of query texts
        if isinstance(input, str):
            input = [input]
        return self.embed_documents(input)


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function using Ollama's native embedding API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=120.0)
    
    def name(self) -> str:
        return "ollama"
    
    def __call__(self, input: Documents) -> Embeddings:
        return self.embed_documents(input)
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of documents using Ollama."""
        if not texts:
            return []
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            try:
                response = self._client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(np.array(data.get("embedding", []), dtype=np.float32))
            except Exception as e:
                print(f"Ollama embedding error: {e}")
                raise
        
        return embeddings
    
    def embed_query(self, input) -> List[np.ndarray]:
        if isinstance(input, str):
            input = [input]
        return self.embed_documents(input)


class EmbeddingServiceFunction(EmbeddingFunction[Documents]):
    """Embedding function using dedicated embedding microservice."""
    
    def __init__(
        self,
        service_url: str = "http://embedding-service:8000",
        model: Optional[str] = None,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=120.0)
    
    def name(self) -> str:
        return "embedding-service"
    
    def __call__(self, input: Documents) -> Embeddings:
        return self.embed_documents(input)
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Embed documents via embedding service."""
        if not texts:
            return []
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            payload = {"texts": texts}
            if self.model:
                payload["model"] = self.model
            
            response = self._client.post(
                f"{self.service_url}/embed",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
        except Exception as e:
            print(f"Embedding service error: {e}")
            raise
    
    def embed_query(self, input) -> List[np.ndarray]:
        if isinstance(input, str):
            input = [input]
        return self.embed_documents(input)


class EmbeddingModel:
    """Wrapper for backward compatibility."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        self._fn = get_embedding_function()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        result = self._fn.embed_query(text)
        return result[0] if result else []


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


@lru_cache(maxsize=1)
def get_embedding_function() -> EmbeddingFunction:
    """Get embedding function based on configuration.
    
    Priority:
    1. EMBEDDING_SERVICE_URL env var -> EmbeddingServiceFunction
    2. LLM_PROVIDER=ollama -> OllamaEmbeddingFunction
    3. Default -> LMStudioEmbeddingFunction
    """
    # Check for dedicated embedding service (microservices mode)
    embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL")
    if embedding_service_url:
        print(f"Using embedding service: {embedding_service_url}")
        return EmbeddingServiceFunction(
            service_url=embedding_service_url,
            model=settings.embedding_model,
        )
    
    # Check LLM provider setting
    provider = settings.llm_provider.lower()
    
    if provider == "ollama":
        print(f"Using Ollama embeddings: {settings.ollama_base_url}")
        return OllamaEmbeddingFunction(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
        )
    
    # Default to LMStudio
    print(f"Using LMStudio embeddings: {settings.lmstudio_base_url}")
    return LMStudioEmbeddingFunction(
        base_url=settings.lmstudio_base_url,
        model=settings.embedding_model,
    )

