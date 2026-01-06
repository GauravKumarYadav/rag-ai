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
from app.db.mysql import get_db_pool, close_db_pool, init_audit_tables
from app.middleware.audit import AuditMiddleware
from app.core.logging import setup_logging, CorrelationIdMiddleware
from app.core.metrics import setup_metrics
from app.evaluation.scheduler import start_evaluation_scheduler, stop_evaluation_scheduler

# Configure structured logging using settings
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
    start_pruning_scheduler()
    
    # Initialize MySQL connection pool and create tables
    try:
        await get_db_pool()
        await init_audit_tables()
        logger.info("MySQL audit logging initialized")
    except Exception as e:
        logger.warning(f"MySQL initialization failed (audit logging disabled): {e}")
    
    # Start evaluation scheduler (runs daily RAG evaluations)
    try:
        start_evaluation_scheduler()
    except Exception as e:
        logger.warning(f"Evaluation scheduler failed to start: {e}")
    
    yield
    
    # Shutdown
    stop_pruning_scheduler()
    stop_evaluation_scheduler()
    
    # Close MySQL connection pool
    try:
        await close_db_pool()
        logger.info("MySQL connection pool closed")
    except Exception as e:
        logger.warning(f"Error closing MySQL pool: {e}")


# OpenAPI tags for better documentation organization
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
]

app = FastAPI(
    title="Local Multimodal Agent",
    version="0.1.0",
    lifespan=lifespan,
    description="""
A local multimodal chatbot with RAG, long-term memory, and vision support.

## Features
- **Multimodal Chat**: Send text and images for analysis
- **RAG**: Retrieve relevant context from uploaded documents
- **Long-term Memory**: Automatic conversation summarization and recall
- **WebSocket Support**: Real-time streaming chat interface
- **Document Management**: Upload, search, and manage knowledge base

## Getting Started
1. Start LMStudio with your preferred model
2. Upload documents via `/documents/upload`
3. Chat via `/chat` (REST) or `/chat/ws/{client_id}` (WebSocket)
""",
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add audit logging middleware (runs after CORS)
app.add_middleware(AuditMiddleware)
app.add_middleware(CorrelationIdMiddleware)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(status.router, tags=["status"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
app.include_router(clients.router, prefix="/clients", tags=["clients"])
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(websocket.router, prefix="/chat", tags=["websocket"])

# Attach Prometheus metrics
setup_metrics(app)

# Serve frontend static files
frontend_path = Path(__file__).parent.parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/admin.html")
async def admin_page():
    """Serve admin dashboard."""
    admin_file = frontend_path / "admin.html"
    if admin_file.exists():
        return FileResponse(admin_file)
    return {"error": "Admin dashboard not found"}


@app.get("/")
async def root() -> dict:
    """Redirect to chat UI or return status."""
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"status": "ok"}

