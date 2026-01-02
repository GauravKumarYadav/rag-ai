from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import chat, clients, conversations, documents, health, models, status, websocket
from app.memory.pruner import start_pruning_scheduler, stop_pruning_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown events."""
    # Startup
    start_pruning_scheduler()
    yield
    # Shutdown
    stop_pruning_scheduler()


# OpenAPI tags for better documentation organization
tags_metadata = [
    {"name": "health", "description": "Health check endpoints"},
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

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(status.router, tags=["status"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
app.include_router(clients.router, prefix="/clients", tags=["clients"])
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(websocket.router, prefix="/chat", tags=["websocket"])


@app.get("/")
async def root() -> dict:
    return {"status": "ok"}

