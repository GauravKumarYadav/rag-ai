# Agentic RAG Chatbot

A multi-agent RAG (Retrieval-Augmented Generation) chatbot powered by LangGraph, LangChain, and LM Studio.

## Features

- **Multi-Agent Architecture**: Specialized agents for query processing, retrieval, and synthesis
- **Client Isolation**: Per-client document collections with global collection support
- **Auto-Summarization**: Automatic conversation summarization when context exceeds threshold
- **Hybrid Search**: BM25 + Vector search with cross-encoder reranking
- **LangSmith Tracing**: Full observability of agent execution
- **Hot Reload**: Documents immediately available after upload
- **WebSocket Support**: Real-time streaming chat

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Docker Compose                     │
├─────────────────────────────────────────────────────┤
│  Redis          │  ChromaDB      │  Chat-API        │
│  (Sessions/     │  (Vector       │  (FastAPI +      │
│   Memory)       │   Store)       │   LangGraph)     │
├─────────────────────────────────────────────────────┤
│  Nginx (Frontend + Reverse Proxy)                   │
└─────────────────────────────────────────────────────┘
        │                    │
        │                    ▼
        │           ┌─────────────────┐
        │           │   LM Studio     │
        │           │   (Local LLM)   │
        │           └─────────────────┘
        │
        ▼
┌─────────────────┐
│   LangSmith     │
│   (Tracing)     │
└─────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- LM Studio running locally with an LLM model loaded
- (Optional) LangSmith account for tracing

### 1. Clone and Configure

```bash
git clone <repository-url>
cd chatbot

# Copy environment file and configure
cp .env.example .env
# Edit .env with your LM Studio URL and model
```

### 2. Start Services

```bash
# Start all services
make up

# Or manually:
docker-compose up -d --build
```

### 3. Access the Application

- **Frontend**: http://localhost
- **API Documentation**: http://localhost:8000/docs
- **ChromaDB**: http://localhost:8020

### 4. Upload Documents

```bash
# Upload a PDF document
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Authorization: Bearer <token>" \
  -F "file=@document.pdf" \
  -F "client_id=global"
```

### 5. Chat

```bash
# REST API
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"message": "What does the document say about X?"}'
```

## Configuration

See `.env.example` for all configuration options.

### Key Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `LLM__LMSTUDIO__BASE_URL` | LM Studio API URL | `http://localhost:1234/v1` |
| `LLM__LMSTUDIO__MODEL` | Model name | `Qwen3-VL-30B-Instruct` |
| `MEMORY__MAX_CONTEXT_TOKENS` | Auto-summarization threshold | `4000` |
| `RAG__RERANKER_ENABLED` | Enable cross-encoder reranking | `true` |
| `RAG__BM25_ENABLED` | Enable hybrid search | `true` |

### LangSmith Integration

To enable LangSmith tracing:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-api-key
LANGCHAIN_PROJECT=agentic-rag
```

## Multi-Agent Flow

```
User Message
     │
     ▼
┌─────────────┐
│ Query Agent │ ← Intent classification, query rewriting
└──────┬──────┘
       │
       ├──[chitchat]──────────────────┐
       │                              │
       ├──[question]──┐               │
       │              ▼               │
       │     ┌───────────────┐        │
       │     │ Retrieval     │        │
       │     │ Agent         │        │
       │     │ (Client+Global)│       │
       │     └───────┬───────┘        │
       │             │                │
       ├──[tool]─────┤                │
       │             ▼                │
       │     ┌───────────────┐        │
       │     │ Tool Agent    │        │
       │     └───────┬───────┘        │
       │             │                │
       └─────────────┼────────────────┘
                     ▼
              ┌─────────────┐
              │ Synthesis   │
              │ Agent       │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Update      │
              │ Memory      │
              └──────┬──────┘
                     │
                     ▼
                 Response
```

## Development

### Local Development

```bash
# Install dependencies
make install

# Start backend in development mode
make dev

# Run tests
make test
```

### Project Structure

```
chatbot/
├── backend/
│   └── app/
│       ├── agents/          # Multi-agent LangGraph system
│       ├── api/routes/      # FastAPI endpoints
│       ├── memory/          # Session buffer & auto-summarization
│       ├── processors/      # Docling document processor
│       ├── rag/             # Vector store & hybrid search
│       └── knowledge/       # Future KG integration
├── frontend/                # Static frontend files
├── docker-compose.yml       # Service orchestration
└── .env                     # Configuration
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send a chat message |
| `/chat/ws/{client_id}` | WS | WebSocket chat |
| `/documents/upload` | POST | Upload documents |
| `/documents` | GET | List documents |
| `/clients` | GET/POST | Manage clients |
| `/conversations` | GET | List conversations |
| `/health` | GET | Health check |

## License

MIT
