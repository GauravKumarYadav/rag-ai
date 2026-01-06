# 🤖 RAG AI Chatbot

A fully local, privacy-focused AI chatbot with RAG (Retrieval-Augmented Generation), long-term memory, vision support, and multi-client data isolation.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.4-orange)
![Ollama](https://img.shields.io/badge/Ollama-0.5+-purple)

## ✨ Features

- 🔒 **100% Local & Private** - All data stays on your machine
- 🤖 **Multi-Provider LLM** - Ollama (recommended), LMStudio, OpenAI, or any OpenAI-compatible API
- 🖼️ **Vision Support** - Analyze images with vision models (llava, qwen2-vl, llama3.2-vision)
- 📚 **RAG Pipeline** - Upload documents (PDF, DOCX, TXT, images) with OCR support
- 🧠 **Long-term Memory** - Automatic conversation summarization and recall
- 👥 **Multi-Client Isolation** - Separate data per client with auto-detection
- 🔄 **Real-time Streaming** - SSE and WebSocket streaming responses
- 📊 **Markdown Tables** - Properly formatted tables in responses
- 💾 **Chat History** - Save, load, and export conversations
- 🔐 **JWT Authentication** - Secure API with token-based auth and MySQL audit logging
- 📈 **Monitoring Stack** - Prometheus metrics, Loki logs, Grafana dashboards
- 🧪 **RAG Evaluation** - Automated quality metrics (precision, recall, MRR, faithfulness)
- 🛡️ **Admin Dashboard** - Audit logs, user management, and system statistics
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
| `LLM_PROVIDER` | `ollama` | `ollama`, `lmstudio`, `openai`, `custom` |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2:1b` | Default chat model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for RAG |
| `MYSQL_HOST` | `mysql` | MySQL host |
| `MYSQL_PORT` | `3306` | MySQL port |
| `MYSQL_DATABASE` | `audit_logs` | Database name |
| `MYSQL_USER` | `root` | Database user |
| `MYSQL_PASSWORD` | `Sarita1!@2024_4` | Database password |
| `REDIS_URL` | `redis://redis:6379` | Redis for session persistence and caching |
| `CHROMADB_HOST` | `chromadb` | ChromaDB host |
| `CHROMADB_PORT` | `8000` | ChromaDB port |

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
│   │   ├── memory/          # Session & long-term memory
│   │   ├── models/          # Pydantic schemas
│   │   ├── rag/             # Vector store & retrieval
│   │   ├── services/        # Business logic
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
│   ├── chroma/              # Vector database storage
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
| `/api/v1/auth/register` | POST | Register new user |
| `/api/v1/auth/login` | POST | Login and get JWT token |
| `/api/v1/auth/me` | GET | Get current user info |

### Core API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Chat with streaming support |
| `/chat/ws/{id}` | WebSocket | Real-time chat |
| `/documents/upload` | POST | Upload documents |
| `/documents/search` | POST | Search documents |
| `/clients` | GET/POST | Manage clients |
| `/clients/{id}` | GET/PUT/DELETE | Client CRUD |
| `/models` | GET | List available models |
| `/models/switch` | POST | Switch LLM provider |
| `/health` | GET | Health check |
| `/stats` | GET | System statistics |
| `/metrics` | GET | Prometheus metrics |

### Admin API (Superuser only)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/admin/audit-logs` | GET | View audit logs |
| `/api/v1/admin/users` | GET/POST | Manage users |
| `/api/v1/admin/stats` | GET | System statistics |
| `/api/v1/admin/config` | GET | View configuration |

### Evaluation API (Superuser only)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/evaluation/generate` | POST | Generate test dataset |
| `/api/v1/evaluation/datasets` | GET | List datasets |
| `/api/v1/evaluation/runs` | POST/GET | Run/list evaluations |

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
