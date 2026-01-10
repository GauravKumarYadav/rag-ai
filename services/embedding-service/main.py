"""
Embedding Service - Dedicated microservice for text embeddings.

Provides a REST API for embedding generation with:
- Multi-provider support (Ollama, LM Studio, or any OpenAI-compatible API)
- Caching (Redis)
- Batching support
- Health checks
- Metrics
"""

import hashlib
import json
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configuration from environment
# EMBEDDING_PROVIDER: "ollama" or "lmstudio" (or any openai-compatible)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

# Ollama settings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# LM Studio settings (OpenAI-compatible)
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://host.containers.internal:1234/v1")

# Common settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
CACHE_TTL = int(os.getenv("CACHE_TTL", "86400"))  # 24 hours

# Global clients
http_client: Optional[httpx.AsyncClient] = None
redis_client: Optional[redis.Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global http_client, redis_client
    
    http_client = httpx.AsyncClient(timeout=120.0)
    
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        print(f"Connected to Redis: {REDIS_URL}")
    except Exception as e:
        print(f"Redis connection failed (caching disabled): {e}")
        redis_client = None
    
    yield
    
    await http_client.aclose()
    if redis_client:
        await redis_client.close()


app = FastAPI(
    title="Embedding Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ============== Request/Response Models ==============

class EmbeddingRequest(BaseModel):
    """Request model for embeddings."""
    texts: List[str]
    model: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""
    embeddings: List[List[float]]
    model: str
    cached: int = 0  # Number of cache hits


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    provider: str
    provider_connected: bool
    redis_connected: bool
    model: str
    provider_url: str


# ============== Helper Functions ==============

def get_cache_key(text: str, model: str) -> str:
    """Generate cache key for embedding."""
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    return f"emb:{model}:{text_hash}"


async def get_cached_embedding(text: str, model: str) -> Optional[List[float]]:
    """Get embedding from cache if available."""
    if not redis_client:
        return None
    
    try:
        key = get_cache_key(text, model)
        cached = await redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


async def cache_embedding(text: str, model: str, embedding: List[float]) -> None:
    """Cache embedding in Redis."""
    if not redis_client:
        return
    
    try:
        key = get_cache_key(text, model)
        await redis_client.setex(key, CACHE_TTL, json.dumps(embedding))
    except Exception:
        pass


async def generate_embedding_ollama(text: str, model: str) -> List[float]:
    """Generate embedding using Ollama API."""
    response = await http_client.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model, "input": text}
    )
    response.raise_for_status()
    data = response.json()
    # Ollama returns embeddings as a list (for batch), get first one
    embeddings = data.get("embeddings", [[]])
    return embeddings[0] if embeddings else []


async def generate_embedding_lmstudio(text: str, model: str) -> List[float]:
    """Generate embedding using LM Studio (OpenAI-compatible) API."""
    response = await http_client.post(
        f"{LMSTUDIO_URL}/embeddings",
        json={"model": model, "input": text}
    )
    response.raise_for_status()
    data = response.json()
    # OpenAI format: data[0].embedding
    if "data" in data and len(data["data"]) > 0:
        return data["data"][0].get("embedding", [])
    return []


async def generate_embedding(text: str, model: str) -> List[float]:
    """Generate embedding using the configured provider."""
    if EMBEDDING_PROVIDER == "lmstudio":
        return await generate_embedding_lmstudio(text, model)
    else:  # Default to ollama
        return await generate_embedding_ollama(text, model)


# ============== API Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    provider_ok = False
    redis_ok = False
    
    # Check provider connectivity
    if EMBEDDING_PROVIDER == "lmstudio":
        provider_url = LMSTUDIO_URL
        try:
            response = await http_client.get(f"{LMSTUDIO_URL}/models")
            provider_ok = response.status_code == 200
        except Exception:
            pass
    else:  # ollama
        provider_url = OLLAMA_URL
        try:
            response = await http_client.get(f"{OLLAMA_URL}/api/tags")
            provider_ok = response.status_code == 200
        except Exception:
            pass
    
    if redis_client:
        try:
            await redis_client.ping()
            redis_ok = True
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if provider_ok else "degraded",
        provider=EMBEDDING_PROVIDER,
        provider_connected=provider_ok,
        redis_connected=redis_ok,
        model=EMBEDDING_MODEL,
        provider_url=provider_url,
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    """Generate embeddings for a list of texts.
    
    Supports caching and batching. Cached embeddings are returned
    immediately, only missing embeddings are generated.
    """
    model = request.model or EMBEDDING_MODEL
    embeddings: List[List[float]] = []
    cache_hits = 0
    
    for text in request.texts:
        # Try cache first
        cached = await get_cached_embedding(text, model)
        if cached:
            embeddings.append(cached)
            cache_hits += 1
            continue
        
        # Generate embedding using configured provider
        try:
            embedding = await generate_embedding(text, model)
            embeddings.append(embedding)
            
            # Cache the result
            await cache_embedding(text, model, embedding)
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=503,
                detail=f"{EMBEDDING_PROVIDER} embedding failed: {str(e)}"
            )
    
    return EmbeddingResponse(
        embeddings=embeddings,
        model=model,
        cached=cache_hits,
    )


@app.get("/models")
async def list_models():
    """List available embedding models from the configured provider."""
    try:
        if EMBEDDING_PROVIDER == "lmstudio":
            response = await http_client.get(f"{LMSTUDIO_URL}/models")
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            return {
                "provider": "lmstudio",
                "models": [m["id"] for m in models]
            }
        else:  # ollama
            response = await http_client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            return {
                "provider": "ollama",
                "models": [m["name"] for m in models]
            }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
