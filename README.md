# 🤖 RAG AI Chatbot

A fully local, privacy-focused AI chatbot with RAG (Retrieval-Augmented Generation), long-term memory, vision support, and multi-client data isolation.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.4-orange)
![Ollama](https://img.shields.io/badge/Ollama-0.5+-purple)

## ✨ Features

- 🔒 **100% Local & Private** - All data stays on your machine
- 🤖 **Multi-Provider LLM** - Ollama, LMStudio (default), OpenAI, or any OpenAI-compatible API
- 🖼️ **Vision Support** - Analyze images with vision models (llava, qwen2-vl, qwen3-vl)
- 📚 **RAG Pipeline** - Upload documents (PDF, DOCX, TXT, images) with OCR support
- 🔍 **Hybrid Search** - BM25 + Vector search with Reciprocal Rank Fusion (RRF)
- 🕸️ **Knowledge Graph** - Per-client entity graphs for query expansion
- ✅ **Answer Verification** - Citation enforcement and grounding checks
- 🧠 **Long-term Memory** - Automatic conversation summarization and recall
- 🤖 **Agentic RAG** - Multi-agent orchestration with query decomposition, multi-hop retrieval, self-correction, and tool calling
- 👥 **Multi-Client Isolation** - Strict client-based data isolation with user-client access control
- 🔄 **Real-time Streaming** - SSE and WebSocket streaming responses
- 📊 **Markdown Tables** - Properly formatted tables in responses
- 💾 **Chat History** - Save, load, and export conversations
- 🔐 **JWT Authentication** - Secure API with token-based auth, user-client assignments, and MySQL audit logging
- 📈 **Monitoring Stack** - Prometheus metrics, Loki logs, Grafana dashboards
- 🧪 **RAG Evaluation** - Automated quality metrics (precision, recall, MRR, faithfulness) + curated gold datasets
- 🛡️ **Admin Dashboard** - Audit logs, user management, client access control, and system statistics
- ⚡ **Pipeline Gating** - Intent-based and confidence-based gating to skip unnecessary stages
- 🐳 **Microservices Architecture** - Scalable containerized deployment with Podman/Docker

## 🏗️ Architecture

The application can run in two modes:

### Standalone Mode
Single process with embedded ChromaDB - good for development and simple deployments.

### Microservices Mode (Recommended for Production)
14 containerized services:

| Service | Port | Description |
|---------|------|-------------|
| **rag-ollama** | 11434 | LLM inference (llama3.2, llava, nomic-embed-text) |
| **rag-chat-api** | 8000 | Main FastAPI backend |
| **rag-embedding-service** | 8010 | Dedicated embedding service with Redis caching |
| **rag-document-api** | 8003 | Document upload and processing API |
| **rag-document-worker** | - | Background document processing |
| **rag-evaluation-worker** | - | Background RAG evaluation |
| **rag-chromadb** | 8020 | Vector database |
| **rag-mysql** | 3307 | User auth and audit logs |
| **rag-redis** | 6379 | Session persistence, embedding cache, and queues |
| **rag-nginx** | 80 | Frontend and reverse proxy |
| **rag-prometheus** | 9090 | Metrics collection |
| **rag-grafana** | 3000 | Monitoring dashboards |
| **rag-loki** | 3100 | Log aggregation |
| **rag-promtail** | - | Log collector |

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (for standalone mode)
- **Podman** or **Docker** (for microservices mode)
- **Git**

### Option A: Microservices Mode (Recommended)

```bash
# Clone the repository
git clone https://github.com/GauravKumarYadav/rag-ai.git
cd rag-ai

# Start all services
podman-compose -f docker-compose.microservices.yml up -d
# or: docker-compose -f docker-compose.microservices.yml up -d

# Pull required Ollama models (first time only)
podman exec rag-ollama ollama pull llama3.2:1b      # Chat model (~1.3GB)
podman exec rag-ollama ollama pull nomic-embed-text  # Embedding model (~274MB)
podman exec rag-ollama ollama pull llava:7b          # Vision model (~4.7GB, optional)

# Access the application
open http://localhost          # Frontend
open http://localhost:8000/docs  # API documentation
open http://localhost:3000     # Grafana dashboards (admin/admin)
```

**Default credentials:** `admin` / `admin123`

### Option B: Standalone Mode (Development)

```bash
# Clone the repository
git clone https://github.com/GauravKumarYadav/rag-ai.git
cd rag-ai

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Install and start Ollama
# Download from https://ollama.ai
ollama pull llama3.2:1b
ollama pull nomic-embed-text
ollama serve  # Runs at http://localhost:11434

# Start MySQL (required for auth)
docker run -d --name mysql -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=root \
  -e MYSQL_DATABASE=chatbot \
  mysql:8.0

# Start the backend
make dev
# or: uvicorn backend.app.main:app --reload --port 8000

# Start the frontend (separate terminal)
make frontend
# or: python -m http.server 3000 -d frontend

# Access the application
open http://localhost:3000
```

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM__PROVIDER` | `lmstudio` | `ollama`, `lmstudio`, `openai`, `custom` |
| `LLM__LMSTUDIO__BASE_URL` | `http://host.containers.internal:1234/v1` | LM Studio API endpoint |
| `LLM__LMSTUDIO__MODEL` | `qwen/qwen3-vl-30b` | Default chat model |
| `LLM__OLLAMA__BASE_URL` | `http://ollama:11434` | Ollama API endpoint |
| `LLM__OLLAMA__MODEL` | `mistral:latest` | Ollama chat model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for RAG |
| `MYSQL_HOST` | `mysql` | MySQL host |
| `MYSQL_PORT` | `3306` | MySQL port |
| `MYSQL_DATABASE` | `audit_logs` | Database name |
| `MYSQL_USER` | `root` | Database user |
| `MYSQL_PASSWORD` | `Sarita1!@2024_4` | Database password |
| `REDIS_URL` | `redis://redis:6379` | Redis for session persistence and caching |
| `CHROMADB_HOST` | `chromadb` | ChromaDB host |
| `CHROMADB_PORT` | `8000` | ChromaDB port |

### RAG Quality Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG__BM25_ENABLED` | `true` | Enable hybrid BM25+vector search |
| `RAG__BM25_WEIGHT` | `0.4` | BM25 weight in RRF fusion |
| `RAG__KNOWLEDGE_GRAPH_ENABLED` | `true` | Enable per-client knowledge graphs |
| `RAG__VERIFICATION_ENABLED` | `true` | Enable answer verification |
| `RAG__CITATION_REQUIRED` | `true` | Require source citations |
| `RAG__MIN_CITATION_COVERAGE` | `0.7` | Min citation coverage (70%) |

### Chunking Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG__CHUNK_TOKEN_SIZE` | `512` | Target chunk size in tokens |
| `RAG__CHUNK_TOKEN_OVERLAP` | `128` | Overlap between chunks |
| `RAG__MIN_CHUNK_TOKENS` | `50` | Minimum chunk size |
| `RAG__RESPECT_HEADINGS` | `true` | Chunk along heading boundaries |

### Pipeline Gating Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG__SKIP_RETRIEVAL_FOR_CHITCHAT` | `true` | Skip RAG for chitchat intents |
| `RAG__CONFIDENCE_GATING_ENABLED` | `true` | Skip stages when confidence is high |
| `RAG__KG_EXPANSION_GATING` | `true` | Only expand KG when initial recall is low |

### Agentic RAG Settings (NEW)

Enable multi-agent orchestration for complex queries:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT__ENABLED` | `false` | Enable agentic pipeline |
| `AGENT__MAX_ITERATIONS` | `3` | Max reasoning iterations per agent |
| `AGENT__MAX_CORRECTIONS` | `2` | Max self-correction attempts |
| `AGENT__MAX_SUB_QUERIES` | `3` | Max query decomposition sub-queries |
| `AGENT__TOOLS_ENABLED` | `true` | Enable tool calling |
| `AGENT__ALLOWED_TOOLS` | `calculator,datetime` | Comma-separated list of allowed tools |
| `AGENT__USE_MODEL_ROUTING` | `true` | Route tasks to appropriate models |

### Vision Models

For image analysis, use vision-capable models:

```bash
# Pull a vision model
podman exec rag-ollama ollama pull llava:7b
# or: ollama pull qwen2-vl:7b

# Send an image in chat
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is in this image?",
    "images": [{"data": "BASE64_IMAGE_DATA", "media_type": "image/png"}]
  }'
## 📁 Project Structure

```
rag-ai/
├── backend/
│   ├── app/
│   │   ├── api/routes/      # API endpoints
│   │   ├── clients/         # LLM client adapters (Ollama, OpenAI, LMStudio)
│   │   ├── evaluation/      # RAG evaluation & quality metrics
│   │   ├── knowledge/       # Knowledge graph (entity extraction, graph store)
│   │   ├── memory/          # Session & long-term memory
│   │   ├── models/          # Pydantic schemas
│   │   ├── rag/             # Vector store, hybrid search, reranking
│   │   ├── services/        # Business logic, verification, compression
│   │   ├── config.py        # Settings
│   │   └── main.py          # FastAPI app
│   ├── Dockerfile
│   └── requirements.txt
├── services/                 # Microservices
│   ├── embedding-service/   # Dedicated embedding API with caching
│   ├── document-api/        # Document upload/processing API
│   ├── document-worker/     # Background document processing
│   └── evaluation-worker/   # Background RAG evaluation
├── frontend/
│   └── index.html           # Single-page chat UI
├── data/
│   ├── chroma/              # Vector database storage (standalone)
│   ├── chroma-docker/       # Vector database storage (microservices)
│   ├── bm25/                # BM25 index storage
│   ├── knowledge_graphs/    # Per-client SQLite knowledge graphs
│   ├── evaluation/          # Evaluation test cases
│   └── raw/                 # Documents to ingest
├── docker/
│   ├── chromadb/            # ChromaDB config
│   ├── nginx/               # Reverse proxy config
│   └── openwebui/           # OpenWebUI alternative frontend
├── docs/                    # Documentation
├── tests/                   # Unit tests
├── docker-compose.yml       # Standalone deployment
├── docker-compose.microservices.yml  # Full microservices deployment
├── .env                     # Environment configuration
├── Makefile                 # Dev commands
└── README.md
```

## 📡 API Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/register` | POST | Register new user |
| `/auth/login` | POST | Login and get JWT token |
| `/auth/me` | GET | Get current user info |

### Core API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Chat with streaming support (include `client_id` for scoped retrieval) |
| `/chat/ws/{id}` | WebSocket | Real-time chat |
| `/documents/upload` | POST | Upload documents (with chunking options) |
| `/documents/search` | POST | Search documents |
| `/documents` | GET | List documents |
| `/documents/{id}` | DELETE | Delete document by ID or filename |
| `/models` | GET | List available models |
| `/models/switch` | POST | Switch LLM provider |
| `/health` | GET | Health check |
| `/status` | GET | System status and model info |
| `/metrics` | GET | Prometheus metrics |

### Client Management API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/clients` | GET/POST | List/create clients |
| `/clients/{id}` | GET/PUT/DELETE | Client CRUD |
| `/clients/{id}/stats` | GET | Get client document stats |
| `/clients/my/assigned` | GET | List current user's accessible clients |
| `/clients/{id}/users` | GET | List users with access to client (admin) |
| `/clients/{id}/users/{user_id}` | POST | Grant user access to client (admin) |
| `/clients/{id}/users/{user_id}` | DELETE | Revoke user access from client (admin) |
| `/clients/user/{user_id}/assigned` | GET | List clients assigned to a user |

### Admin API (Superuser only)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/audit-logs` | GET | View audit logs |
| `/admin/users` | GET/POST | Manage users |
| `/admin/users/{id}` | PATCH/DELETE | Update/delete user |
| `/admin/stats` | GET | System statistics |
| `/admin/config` | GET | View configuration |

### Evaluation API (Superuser only)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/evaluation/datasets` | POST/GET | Generate/list synthetic datasets |
| `/evaluation/runs` | POST/GET | Run/list evaluations |
| `/evaluation/runs/{id}` | GET | Get evaluation details |
| `/evaluation/curated-datasets` | GET | List curated gold datasets |
| `/evaluation/curated-runs` | POST | Run curated evaluation |
| `/evaluation/curated-runs/{path}/details` | GET | Get curated evaluation details |

## 👥 Client Isolation & Access Control

### Overview

The system enforces strict client-based data isolation:
- Each **client** has its own documents, knowledge graph, and memories
- **Users** must be assigned to clients to access their data
- **Admins** can access all clients and manage user assignments

### Managing User Access (Admin)

```bash
# Assign user to a client
curl -X POST http://localhost:8000/clients/{client_id}/users/{user_id} \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Revoke user access
curl -X DELETE http://localhost:8000/clients/{client_id}/users/{user_id} \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# List users with access to a client
curl http://localhost:8000/clients/{client_id}/users \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Using the UI

1. Go to **Profile → Clients** tab
2. In the **User Access Management** panel:
   - Select a client
   - Select a user
   - Click **Grant Access** or **Revoke Access**

### Chat with Client Context

```bash
# Include client_id in chat requests for scoped retrieval
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the policies?",
    "client_id": "your-client-uuid"
  }'
```

In the UI, select a client from the header dropdown before chatting.

## 🐳 Docker Deployment

```bash
# Start all services
make docker-up

# View logs
docker-compose logs -f

# Stop services
make docker-down
```

## 🧪 Running Tests

```bash
make test

# Or with coverage
pytest tests/ -v --cov=backend/app
```

## 📝 Available Make Commands

```bash
make help       # Show all commands
make install    # Install dependencies
make dev        # Start backend server
make frontend   # Start frontend server
make ingest     # Ingest documents from data/raw
make test       # Run tests
make docker-up  # Start with Docker
make docker-down # Stop Docker
make lint       # Check for syntax errors
make clean      # Clean generated files
```

## 🔄 Switching LLM Providers at Runtime

```bash
# Switch to Ollama
curl -X POST http://localhost:8000/models/switch \
  -H "Content-Type: application/json" \
  -d '{"provider": "ollama", "model": "llama3"}'

# Switch to OpenAI
curl -X POST http://localhost:8000/models/switch \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "model": "gpt-4", "api_key": "sk-..."}'
```

## 🛠️ Troubleshooting

### LM Studio Connection Failed
- Ensure LM Studio is running on the host machine
- Check the model is loaded (e.g., `qwen/qwen3-vl-30b`)
- For containers, use `http://host.containers.internal:1234/v1` as base URL
- Verify `.env` settings:
  ```bash
  LLM__PROVIDER=lmstudio
  LLM__LMSTUDIO__BASE_URL=http://host.containers.internal:1234/v1
  LLM__LMSTUDIO__MODEL=qwen/qwen3-vl-30b
  ```

### Ollama Connection Failed
- Check Ollama is running: `podman logs rag-ollama` or `ollama list`
- Verify models are pulled: `podman exec rag-ollama ollama list`
- Check network connectivity: `curl http://localhost:11434/api/tags`

### Embedding Errors (503)
- Ensure `nomic-embed-text` model is pulled
- Check embedding service logs: `podman logs rag-embedding-service`
- Verify Redis is running: `podman logs rag-redis`

### Chat Returns 400 Error
- For vision models, ensure images are base64 encoded without the `data:` prefix
- Check the model supports vision (llava, qwen2-vl, llama3.2-vision)
- Review chat-api logs: `podman logs rag-chat-api`

### Authentication/Login Issues
- Default credentials: `admin` / `admin123`
- Check MySQL is running: `podman logs rag-mysql`
- Verify CORS origins include your frontend URL

### ChromaDB Issues
- See [docs/CHROMADB.md](docs/CHROMADB.md) for detailed setup
- Check data directory permissions: `chmod -R 755 ./data/chroma`
- Verify ChromaDB v2 API: `curl http://localhost:8020/api/v2/heartbeat`

### Container Issues
```bash
# Check all container statuses
podman ps --format "table {{.Names}}\t{{.Status}}"

# View logs for a specific service
podman logs rag-chat-api

# Restart all services
podman-compose -f docker-compose.microservices.yml down
podman-compose -f docker-compose.microservices.yml up -d
```

## 📚 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System architecture and components
- [Setup Guide](docs/SETUP.md) - Quick start instructions
- [ChromaDB Guide](docs/CHROMADB.md) - Vector database setup and configuration
- [Monitoring Guide](docs/MONITORING.md) - Prometheus, Loki, Grafana setup
- [Evaluation Guide](docs/EVALUATION.md) - RAG evaluation framework
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)

## 📄 License

MIT License - Feel free to use and modify.

## 🙏 Acknowledgments

- [LMStudio](https://lmstudio.ai) - Local LLM inference
- [ChromaDB](https://www.trychroma.com) - Vector database
- [FastAPI](https://fastapi.tiangolo.com) - Backend framework
- [Docling](https://github.com/DS4SD/docling) - Document parsing with OCR
