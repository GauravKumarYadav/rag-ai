"""
Document Worker - Background worker for document processing.

Consumes document processing jobs from Redis queue:
- PDF text extraction
- DOCX parsing  
- Image OCR
- Text chunking
- Embedding generation
- ChromaDB storage
"""

import asyncio
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime
from typing import List, Optional, Dict, Any

import httpx
import redis.asyncio as redis
from pypdf import PdfReader
from docx import Document as DocxDocument
from PIL import Image
import pytesseract

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
CHROMADB_URL = os.getenv("CHROMADB_URL", "http://chromadb:8000")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8000")
QUEUE_NAME = os.getenv("QUEUE_NAME", "document_jobs")
WORKER_ID = os.getenv("WORKER_ID", "worker-1")

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


class DocumentWorker:
    """Background worker for document processing."""
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.http: Optional[httpx.AsyncClient] = None
        self.running = False
    
    async def start(self):
        """Initialize connections and start processing."""
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self.http = httpx.AsyncClient(timeout=120.0)
        self.running = True
        
        print(f"[{WORKER_ID}] Document worker started")
        print(f"[{WORKER_ID}] Redis: {REDIS_URL}")
        print(f"[{WORKER_ID}] ChromaDB: {CHROMADB_URL}")
        print(f"[{WORKER_ID}] Embedding Service: {EMBEDDING_SERVICE_URL}")
        
        await self.process_loop()
    
    async def stop(self):
        """Cleanup and stop worker."""
        self.running = False
        if self.http:
            await self.http.aclose()
        if self.redis:
            await self.redis.close()
        print(f"[{WORKER_ID}] Document worker stopped")
    
    async def process_loop(self):
        """Main processing loop - consume jobs from Redis queue."""
        while self.running:
            try:
                # Block waiting for job (timeout 5s to allow graceful shutdown)
                job = await self.redis.brpop(QUEUE_NAME, timeout=5)
                
                if job:
                    _, job_data = job
                    await self.process_job(json.loads(job_data))
                    
            except Exception as e:
                print(f"[{WORKER_ID}] Error in process loop: {e}")
                await asyncio.sleep(1)
    
    async def process_job(self, job: Dict[str, Any]):
        """Process a single document job."""
        job_id = job.get("job_id", "unknown")
        filename = job.get("filename", "unknown")
        client_id = job.get("client_id")
        
        print(f"[{WORKER_ID}] Processing job {job_id}: {filename}")
        
        try:
            # Update status to processing
            await self.update_status(job_id, "processing")
            
            # Get document content from Redis (stored as binary)
            content_key = f"doc_content:{job_id}"
            content_b64 = await self.redis.get(content_key)
            
            if not content_b64:
                raise ValueError("Document content not found in Redis")
            
            import base64
            content = base64.b64decode(content_b64)
            
            # Extract text based on file type
            text = await self.extract_text(content, filename)
            
            if not text.strip():
                raise ValueError("No text could be extracted from document")
            
            # Chunk the text
            chunks = self.chunk_text(text)
            print(f"[{WORKER_ID}] Extracted {len(chunks)} chunks from {filename}")
            
            # Generate embeddings
            embeddings = await self.generate_embeddings([c["text"] for c in chunks])
            
            # Store in ChromaDB
            await self.store_in_chromadb(
                chunks=chunks,
                embeddings=embeddings,
                filename=filename,
                client_id=client_id,
                job_id=job_id,
            )
            
            # Cleanup content from Redis
            await self.redis.delete(content_key)
            
            # Update status to completed
            await self.update_status(job_id, "completed", {
                "chunks": len(chunks),
                "filename": filename,
            })
            
            print(f"[{WORKER_ID}] Job {job_id} completed: {len(chunks)} chunks stored")
            
        except Exception as e:
            print(f"[{WORKER_ID}] Job {job_id} failed: {e}")
            await self.update_status(job_id, "failed", {"error": str(e)})
    
    async def update_status(self, job_id: str, status: str, metadata: Optional[Dict] = None):
        """Update job status in Redis."""
        status_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat(),
            "worker_id": WORKER_ID,
        }
        if metadata:
            status_data.update(metadata)
        
        await self.redis.hset(f"job_status:{job_id}", mapping=status_data)
        await self.redis.expire(f"job_status:{job_id}", 86400)  # 24h TTL
    
    async def extract_text(self, content: bytes, filename: str) -> str:
        """Extract text from document based on file type."""
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        
        if ext == "pdf":
            return self.extract_pdf(content)
        elif ext in ("docx", "doc"):
            return self.extract_docx(content)
        elif ext in ("png", "jpg", "jpeg", "gif", "bmp", "tiff"):
            return self.extract_image(content)
        elif ext in ("txt", "md", "markdown", "rst", "csv", "json"):
            return content.decode("utf-8", errors="ignore")
        else:
            # Try as text
            return content.decode("utf-8", errors="ignore")
    
    def extract_pdf(self, content: bytes) -> str:
        """Extract text from PDF."""
        reader = PdfReader(io.BytesIO(content))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    
    def extract_docx(self, content: bytes) -> str:
        """Extract text from DOCX."""
        doc = DocxDocument(io.BytesIO(content))
        return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
    
    def extract_image(self, content: bytes) -> str:
        """Extract text from image using OCR."""
        image = Image.open(io.BytesIO(content))
        return pytesseract.image_to_string(image)
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Generate stable chunk ID
                chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:8]
                chunks.append({
                    "text": chunk_text,
                    "chunk_idx": chunk_idx,
                    "start_char": start,
                    "end_char": end,
                    "chunk_id": f"chunk_{chunk_idx}_{chunk_hash}",
                })
                chunk_idx += 1
            
            start = end - CHUNK_OVERLAP
            if start >= len(text) - CHUNK_OVERLAP:
                break
        
        return chunks
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via embedding service."""
        response = await self.http.post(
            f"{EMBEDDING_SERVICE_URL}/embed",
            json={"texts": texts}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embeddings", [])
    
    async def store_in_chromadb(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
        filename: str,
        client_id: Optional[str],
        job_id: str,
    ):
        """Store chunks with embeddings in ChromaDB."""
        # Determine collection name
        if client_id:
            collection_name = f"{client_id}_documents"
        else:
            collection_name = "documents"
        
        # Ensure collection exists
        try:
            await self.http.post(
                f"{CHROMADB_URL}/api/v2/collections",
                json={"name": collection_name, "get_or_create": True}
            )
        except Exception:
            pass  # Collection may already exist
        
        # Get collection
        response = await self.http.get(
            f"{CHROMADB_URL}/api/v2/collections/{collection_name}"
        )
        
        if response.status_code == 404:
            # Create collection
            response = await self.http.post(
                f"{CHROMADB_URL}/api/v2/collections",
                json={"name": collection_name}
            )
            response.raise_for_status()
        
        collection = response.json()
        collection_id = collection.get("id", collection_name)
        
        # Prepare data for upsert
        ids = [f"{filename}#{c['chunk_id']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": filename,
                "chunk_idx": c["chunk_idx"],
                "start_char": c["start_char"],
                "end_char": c["end_char"],
                "job_id": job_id,
                "client_id": client_id or "",
            }
            for c in chunks
        ]
        
        # Add to collection
        response = await self.http.post(
            f"{CHROMADB_URL}/api/v2/collections/{collection_id}/add",
            json={
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas,
            }
        )
        response.raise_for_status()


async def main():
    """Main entry point."""
    worker = DocumentWorker()
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        pass
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
