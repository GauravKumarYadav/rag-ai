"""
Main FastAPI application.

Simplified multi-agent RAG chatbot with:
- LangGraph multi-agent architecture
- Redis-persisted memory with auto-summarization
- ChromaDB vector store with global + client collections
- LM Studio for LLM inference
- LangSmith for tracing
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.api.routes import (
    admin,
    auth,
    chat,
    clients,
    conversations,
    documents,
    evaluation,
    health,
    models,
    status,
    websocket,
)
from app.memory.pruner import start_pruning_scheduler, stop_pruning_scheduler
from app.core.logging import setup_logging, CorrelationIdMiddleware

# Configure structured logging
setup_logging(
    log_level=settings.logging.level,
    log_file=settings.logging.log_file,
    log_dir=settings.logging.log_dir,
    max_bytes=settings.logging.max_bytes,
    backup_count=settings.logging.backup_count,
    json_format=settings.logging.json_format,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown events."""
    # Startup
    logger.info("Starting RAG Chatbot API...")
    
    # Start memory pruning scheduler
    start_pruning_scheduler()
    logger.info("Memory pruning scheduler started")
    
    # Initialize Redis connection (via session buffer)
    try:
        from app.memory.session_buffer import get_session_buffer
        buffer = get_session_buffer()
        logger.info("Session buffer initialized (Redis connection established)")
    except Exception as e:
        logger.warning(f"Redis connection failed, using in-memory storage: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Chatbot API...")
    stop_pruning_scheduler()
    logger.info("Memory pruning scheduler stopped")


# OpenAPI tags for documentation
tags_metadata = [
    {"name": "health", "description": "Health check endpoints"},
    {"name": "auth", "description": "Authentication endpoints for login and user info"},
    {"name": "chat", "description": "Chat and conversation endpoints with multimodal support"},
    {"name": "websocket", "description": "Real-time WebSocket chat interface"},
    {"name": "documents", "description": "Document upload, search, and management"},
    {"name": "conversations", "description": "Conversation history and memory management"},
    {"name": "clients", "description": "Client management and data isolation"},
    {"name": "models", "description": "LLM model information and configuration"},
    {"name": "status", "description": "System status and statistics"},
    {"name": "admin", "description": "Admin endpoints for user management"},
]

app = FastAPI(
    title="Agentic RAG Chatbot",
    version="2.0.0",
    lifespan=lifespan,
    description="""
A multi-agent RAG chatbot with LangGraph orchestration.

## Features
- **Multi-Agent Architecture**: Specialized agents for query processing, retrieval, and synthesis
- **Client Isolation**: Per-client document collections with global collection support
- **Auto-Summarization**: Automatic conversation summarization when context exceeds threshold
- **Hybrid Search**: BM25 + Vector search with cross-encoder reranking
- **LangSmith Tracing**: Full observability of agent execution
- **Hot Reload**: Documents immediately available after upload

## Getting Started
1. Start LM Studio with your preferred model
2. Start the Docker stack: `docker-compose up -d`
3. Upload documents via `/documents/upload`
4. Chat via `/chat` (REST) or `/chat/ws/{client_id}` (WebSocket)
""",
    openapi_tags=tags_metadata,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Correlation ID middleware for request tracing
app.add_middleware(CorrelationIdMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(status.router, tags=["status"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
app.include_router(clients.router, prefix="/clients", tags=["clients"])
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])
app.include_router(websocket.router, prefix="/chat", tags=["websocket"])

# Serve frontend static files
frontend_path = Path(__file__).parent.parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
async def root() -> dict:
    """Serve chat UI or return status."""
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"status": "ok", "version": "2.0.0"}
