"""
Embedding functions for vector stores.

Supports multiple embedding providers:
- LMStudio (OpenAI-compatible API)
- Ollama (native embeddings API)
- Embedding Service (microservices mode)
"""

import os
from functools import lru_cache
from typing import List, Optional

import httpx
from chromadb import EmbeddingFunction, Documents, Embeddings

from app.config import settings


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
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
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
            return [item["embedding"] for item in embeddings]
        except Exception as e:
            print(f"Embedding error for texts: {texts[:100] if texts else texts}, error: {e}")
            raise
    
    def embed_query(self, input) -> Embeddings:
        """Embed query texts. ChromaDB expects List[List[float]] for queries."""
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
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
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
                embeddings.append(data.get("embedding", []))
            except Exception as e:
                print(f"Ollama embedding error: {e}")
                raise
        
        return embeddings
    
    def embed_query(self, input) -> Embeddings:
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
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
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
            return data.get("embeddings", [])
        except Exception as e:
            print(f"Embedding service error: {e}")
            raise
    
    def embed_query(self, input) -> Embeddings:
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

