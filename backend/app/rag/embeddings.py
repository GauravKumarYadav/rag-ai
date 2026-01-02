from functools import lru_cache
from typing import List, Optional

import httpx
from chromadb import EmbeddingFunction, Documents, Embeddings

from app.config import settings


class LMStudioEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function using LMStudio's local embedding API."""
    
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


class EmbeddingModel:
    """Wrapper for backward compatibility."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        # Use LMStudio embeddings - model_name is ignored, uses LMStudio's loaded model
        self._fn = LMStudioEmbeddingFunction(
            base_url=settings.lmstudio_base_url,
            model=settings.embedding_model,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._fn.embed_query(text)


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


@lru_cache(maxsize=1)
def get_embedding_function() -> LMStudioEmbeddingFunction:
    """Get embedding function for ChromaDB."""
    return LMStudioEmbeddingFunction(
        base_url=settings.lmstudio_base_url,
        model=settings.embedding_model,
    )

