# Architecture Overview

This document describes the overall system architecture, key modules, and data flow for the RAG AI Chatbot.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Frontend (Vanilla HTML/JS)                      │
│   index.html (Chat UI)  │  admin.html (Admin Dashboard)                     │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ HTTP / WebSocket
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             FastAPI Backend                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   Auth   │  │   Chat   │  │Documents │  │  Admin   │  │Evaluation│      │
│  │  Routes  │  │  Routes  │  │  Routes  │  │  Routes  │  │  Routes  │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                       Middleware Layer                              │    │
│  │  CORS │ Audit Logging │ Correlation ID │ Prometheus Metrics        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                      Core Services                                  │    │
│  │  ChatService │ PromptBuilder │ ClientExtractor │ SessionBuffer     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              ▼                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  LLM Client │  │ RAG/Vector  │  │  Memory     │  │ Evaluation  │        │
│  │  (LMStudio) │  │  Store      │  │  (Long-term)│  │  Framework  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
┌──────────────┐          ┌──────────────┐              ┌──────────────┐
│   ChromaDB   │          │    MySQL     │              │   LMStudio   │
│ (Embeddings) │          │ (Audit Logs) │              │   (LLM API)  │
└──────────────┘          └──────────────┘              └──────────────┘
```

---

## Module Breakdown

### Authentication (`backend/app/auth/`)
- **JWT-based authentication** with bcrypt password hashing
- `dependencies.py` - `get_current_user`, `require_superuser`
- `jwt.py` - Token creation/validation
- `password.py` - Password hashing utilities

### API Routes (`backend/app/api/routes/`)
| Route File | Prefix | Description |
|------------|--------|-------------|
| `auth.py` | `/auth` | Login, logout, user info |
| `chat.py` | `/chat` | Chat completions (REST + streaming) |
| `websocket.py` | `/chat/ws` | WebSocket chat interface |
| `documents.py` | `/documents` | Upload, search, stats |
| `conversations.py` | `/conversations` | History, memory |
| `clients.py` | `/clients` | Multi-client management |
| `admin.py` | `/admin` | Audit logs, users, stats, config |
| `evaluation.py` | `/evaluation` | RAG evaluation datasets & runs |
| `health.py` | `/health` | Health checks |
| `status.py` | `/status` | System status |
| `models.py` | `/models` | LLM model info & switching |

### Core (`backend/app/core/`)
- `logging.py` - Structured JSON logging with correlation IDs, file rotation
- `metrics.py` - Prometheus instrumentation with custom RAG/LLM histograms

### RAG Pipeline (`backend/app/rag/`)
- `vector_store.py` - ChromaDB integration for documents & memories
- `retriever.py` - Semantic search with metadata filtering
- `embeddings.py` - LMStudio embedding function
- `factory.py` - Vector store factory (supports ChromaDB, Pinecone, etc.)

### Memory (`backend/app/memory/`)
- `session_buffer.py` - In-memory conversation buffers
- `long_term.py` - Summarization and long-term storage
- `pruner.py` - Background cleanup scheduler

### Evaluation (`backend/app/evaluation/`)
- `generator.py` - Auto-generate Q&A pairs from documents using LLM
- `metrics.py` - Precision@K, Recall@K, MRR, Faithfulness
- `runner.py` - Execute evaluations and store results
- `scheduler.py` - APScheduler for cron-based daily runs

### Database (`backend/app/db/`)
- `mysql.py` - Connection pool, audit logging, user queries

### Document Processing (`backend/app/processors/`)
- `registry.py` - Plugin registry for document processors
- Supports PDF, DOCX, TXT, images with OCR

---

## Data Flow

### Chat Request Flow
```
1. User sends message (REST or WebSocket)
2. Auth middleware validates JWT
3. Correlation ID middleware adds request tracing
4. ChatService receives request
5. PromptBuilder constructs system prompt + context
6. RAG retriever fetches relevant document chunks
7. LLM client streams response
8. Session buffer stores conversation
9. Audit middleware logs request
10. Response streamed to user
```

### Document Upload Flow
```
1. User uploads file(s) with client_name
2. ProcessorRegistry selects appropriate processor
3. Text extracted (with OCR if needed)
4. Text chunked with overlap
5. Chunks embedded via LMStudio
6. Chunks stored in ChromaDB with metadata
7. Response returned with chunk count
```

### Evaluation Flow
```
1. Admin generates dataset from documents
2. LLM creates Q&A pairs from random chunks
3. Dataset stored in MySQL
4. Admin triggers evaluation run
5. For each Q&A: retrieve docs, compute metrics
6. Aggregated results stored in MySQL
7. Results viewable in admin dashboard
```

---

## Configuration

Settings are managed via `backend/app/config.py` with nested Pydantic models:

| Group | Key Settings |
|-------|--------------|
| `llm` | provider, model, temperature, timeout |
| `rag` | provider, chroma_db_path, embedding_model |
| `mysql` | host, port, database, user, password |
| `logging` | level, log_dir, json_format, retention |
| `evaluation` | sample_size, cron_schedule, timezone |
| `jwt` | secret_key, algorithm, expire_minutes |

---

## Observability Stack

| Component | Purpose | Port |
|-----------|---------|------|
| Prometheus | Metrics collection | 9090 |
| Loki | Log aggregation | 3100 |
| Promtail | Log shipping | - |
| Grafana | Dashboards | 3001 |

Start with:
```bash
cd docker/monitoring
podman-compose up -d
```

---

## Security

- JWT tokens for API authentication
- bcrypt password hashing (12 rounds)
- Superuser role for admin endpoints
- Audit logging of all API requests
- CORS configured for allowed origins
- Credentials stored in environment variables
