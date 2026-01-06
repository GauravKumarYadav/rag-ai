"""
Document API - REST API for document upload and management.

Handles document uploads, enqueues processing jobs, and tracks status.
Delegates heavy processing to document-worker.
"""

import base64
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
CHROMADB_URL = os.getenv("CHROMADB_URL", "http://chromadb:8000")
QUEUE_NAME = os.getenv("QUEUE_NAME", "document_jobs")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024)))  # 50MB

# Allowed file types
ALLOWED_EXTENSIONS = {
    "pdf", "docx", "doc", "txt", "md", "markdown",
    "png", "jpg", "jpeg", "gif", "bmp", "tiff",
    "csv", "json", "rst",
}

# Global clients
redis_client: Optional[redis.Redis] = None
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global redis_client, http_client
    
    redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    http_client = httpx.AsyncClient(timeout=30.0)
    
    await redis_client.ping()
    print(f"Connected to Redis: {REDIS_URL}")
    
    yield
    
    await redis_client.close()
    await http_client.aclose()


app = FastAPI(
    title="Document API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Request/Response Models ==============

class UploadResponse(BaseModel):
    """Response for document upload."""
    job_id: str
    filename: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Status of a processing job."""
    job_id: str
    status: str
    updated_at: Optional[str] = None
    worker_id: Optional[str] = None
    chunks: Optional[int] = None
    error: Optional[str] = None


class DocumentInfo(BaseModel):
    """Information about a stored document."""
    id: str
    source: str
    chunk_idx: int
    content_preview: str


class SearchRequest(BaseModel):
    """Request for document search."""
    query: str
    top_k: int = 5
    client_id: Optional[str] = None


class SearchResult(BaseModel):
    """Search result."""
    id: str
    content: str
    score: float
    metadata: dict


# ============== Helper Functions ==============

def get_file_extension(filename: str) -> str:
    """Get lowercase file extension."""
    return filename.lower().split(".")[-1] if "." in filename else ""


def validate_file(filename: str, size: int) -> None:
    """Validate file before processing."""
    ext = get_file_extension(filename)
    
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )


# ============== API Endpoints ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    redis_ok = False
    chromadb_ok = False
    
    try:
        await redis_client.ping()
        redis_ok = True
    except Exception:
        pass
    
    try:
        response = await http_client.get(f"{CHROMADB_URL}/api/v2/heartbeat")
        chromadb_ok = response.status_code == 200
    except Exception:
        pass
    
    return {
        "status": "healthy" if (redis_ok and chromadb_ok) else "degraded",
        "redis_connected": redis_ok,
        "chromadb_connected": chromadb_ok,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    client_id: Optional[str] = Form(None),
):
    """Upload a document for processing.
    
    The document is stored in Redis and a job is queued for background processing.
    Use the job_id to check processing status.
    """
    # Read file content
    content = await file.read()
    filename = file.filename or "unknown"
    
    # Validate
    validate_file(filename, len(content))
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Store content in Redis (base64 encoded)
    content_key = f"doc_content:{job_id}"
    await redis_client.setex(content_key, 3600, base64.b64encode(content))  # 1h TTL
    
    # Create job
    job_data = {
        "job_id": job_id,
        "filename": filename,
        "client_id": client_id,
        "created_at": datetime.utcnow().isoformat(),
        "size_bytes": len(content),
    }
    
    # Initialize job status
    await redis_client.hset(
        f"job_status:{job_id}",
        mapping={
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "filename": filename,
        }
    )
    await redis_client.expire(f"job_status:{job_id}", 86400)
    
    # Enqueue job
    import json
    await redis_client.lpush(QUEUE_NAME, json.dumps(job_data))
    
    return UploadResponse(
        job_id=job_id,
        filename=filename,
        status="queued",
        message="Document queued for processing",
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a document processing job."""
    status_data = await redis_client.hgetall(f"job_status:{job_id}")
    
    if not status_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Decode bytes to str
    status_dict = {
        k.decode() if isinstance(k, bytes) else k: 
        v.decode() if isinstance(v, bytes) else v
        for k, v in status_data.items()
    }
    
    return JobStatus(
        job_id=job_id,
        status=status_dict.get("status", "unknown"),
        updated_at=status_dict.get("updated_at"),
        worker_id=status_dict.get("worker_id"),
        chunks=int(status_dict["chunks"]) if "chunks" in status_dict else None,
        error=status_dict.get("error"),
    )


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents(client_id: Optional[str] = None, limit: int = 100):
    """List documents stored in ChromaDB."""
    collection_name = f"{client_id}_documents" if client_id else "documents"
    
    try:
        # Get collection
        response = await http_client.get(
            f"{CHROMADB_URL}/api/v2/collections/{collection_name}"
        )
        
        if response.status_code == 404:
            return []
        
        collection = response.json()
        collection_id = collection.get("id", collection_name)
        
        # Get documents
        response = await http_client.post(
            f"{CHROMADB_URL}/api/v2/collections/{collection_id}/get",
            json={"limit": limit, "include": ["metadatas", "documents"]}
        )
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        
        documents = []
        for i, doc_id in enumerate(data.get("ids", [])):
            metadata = data.get("metadatas", [{}])[i] or {}
            content = data.get("documents", [""])[i] or ""
            
            documents.append(DocumentInfo(
                id=doc_id,
                source=metadata.get("source", "unknown"),
                chunk_idx=metadata.get("chunk_idx", 0),
                content_preview=content[:200] + "..." if len(content) > 200 else content,
            ))
        
        return documents
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """Search documents using vector similarity."""
    collection_name = f"{request.client_id}_documents" if request.client_id else "documents"
    
    try:
        # Get embeddings for query
        embed_response = await http_client.post(
            f"{os.getenv('EMBEDDING_SERVICE_URL', 'http://embedding-service:8000')}/embed",
            json={"texts": [request.query]}
        )
        embed_response.raise_for_status()
        query_embedding = embed_response.json()["embeddings"][0]
        
        # Get collection
        response = await http_client.get(
            f"{CHROMADB_URL}/api/v2/collections/{collection_name}"
        )
        
        if response.status_code == 404:
            return []
        
        collection = response.json()
        collection_id = collection.get("id", collection_name)
        
        # Query
        response = await http_client.post(
            f"{CHROMADB_URL}/api/v2/collections/{collection_id}/query",
            json={
                "query_embeddings": [query_embedding],
                "n_results": request.top_k,
                "include": ["documents", "metadatas", "distances"],
            }
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        ids = data.get("ids", [[]])[0]
        documents = data.get("documents", [[]])[0]
        metadatas = data.get("metadatas", [[]])[0]
        distances = data.get("distances", [[]])[0]
        
        for i, doc_id in enumerate(ids):
            results.append(SearchResult(
                id=doc_id,
                content=documents[i] if i < len(documents) else "",
                score=distances[i] if i < len(distances) else 0.0,
                metadata=metadatas[i] if i < len(metadatas) else {},
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, client_id: Optional[str] = None):
    """Delete a document from ChromaDB."""
    collection_name = f"{client_id}_documents" if client_id else "documents"
    
    try:
        response = await http_client.get(
            f"{CHROMADB_URL}/api/v2/collections/{collection_name}"
        )
        
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        collection = response.json()
        collection_id = collection.get("id", collection_name)
        
        response = await http_client.post(
            f"{CHROMADB_URL}/api/v2/collections/{collection_id}/delete",
            json={"ids": [document_id]}
        )
        response.raise_for_status()
        
        return {"message": "Document deleted", "id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
