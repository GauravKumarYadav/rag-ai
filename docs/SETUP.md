# Local Multimodal AI Chatbot - Setup Guide

A fully local AI chatbot with RAG, long-term memory, vision support, and multi-provider LLM backend.

## Features

- 🤖 **Multi-Provider LLM**: Ollama (recommended), LMStudio, OpenAI, or any OpenAI-compatible endpoint
- 📚 **RAG**: Upload documents and query them with semantic search
- 🧠 **Long-term Memory**: Automatic conversation summarization and recall
- 🖼️ **Vision Support**: Send images for analysis with vision models (llava, qwen2-vl)
- 🔄 **Real-time Streaming**: WebSocket and SSE streaming responses
- 🎨 **Modern Frontend**: Clean chat interface with settings panel
- 🔌 **Modular Vector Store**: ChromaDB with Redis-cached embeddings
- 🔐 **JWT Authentication**: Secure API with token-based authentication
- 📈 **Monitoring Stack**: Prometheus metrics, Loki logs, Grafana dashboards
- 🧪 **RAG Evaluation**: Automated quality testing with precision/recall metrics
- 🛡️ **Admin Dashboard**: Audit logs, user management, and system statistics
- 🐳 **Microservices Architecture**: Scalable containerized deployment

## Prerequisites

- **Podman** or **Docker** (for microservices deployment)
- **Git**
- **Python 3.11+** (optional, for standalone mode)

---

## Quick Start (Microservices Mode - Recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/GauravKumarYadav/rag-ai.git
cd rag-ai
```

### 2. Start All Services

```bash
# Using Podman (recommended)
podman-compose -f docker-compose.microservices.yml up -d

# Or using Docker
docker-compose -f docker-compose.microservices.yml up -d
```

This starts 14 services:
- **rag-ollama** (port 11434) - LLM inference
- **rag-chat-api** (port 8000) - Main API
- **rag-embedding-service** (port 8010) - Embedding with caching
- **rag-document-api** (port 8003) - Document processing
- **rag-chromadb** (port 8020) - Vector database
- **rag-mysql** (port 3307) - User auth & audit logs
- **rag-redis** (port 6379) - Caching
- **rag-nginx** (port 80) - Frontend & reverse proxy
- **rag-grafana** (port 3000) - Monitoring dashboards
- **rag-prometheus** (port 9090) - Metrics
- **rag-loki** (port 3100) - Log aggregation
- **rag-promtail** - Log collector
- **rag-document-worker** - Background processing
- **rag-evaluation-worker** - RAG evaluation

### 3. Pull Required Ollama Models

```bash
# Chat model (required, ~1.3GB)
podman exec rag-ollama ollama pull llama3.2:1b

# Embedding model (required, ~274MB)
podman exec rag-ollama ollama pull nomic-embed-text

# Vision model (optional, ~4.7GB)
podman exec rag-ollama ollama pull llava:7b
```

### 4. Verify Services

```bash
# Check all containers are running
podman ps --format "table {{.Names}}\t{{.Status}}"

# Test the chat API
TOKEN=$(curl -s http://localhost:8000/auth/login -X POST \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')

curl -s http://localhost:8000/chat -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello!","stream":false}' | jq '.response'
```

### 5. Access the Application

- **Frontend**: http://localhost
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

**Default credentials**: `admin` / `admin123`

---

## Standalone Mode (Development)

For development without containers:
---

## Standalone Mode (Development)

For development without containers:

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Install and Start Ollama

```bash
# Install from https://ollama.ai
ollama pull llama3.2:1b
ollama pull nomic-embed-text
ollama serve  # Runs at http://localhost:11434
```

### 3. Start MySQL

```bash
# Using Docker
docker run -d --name mysql -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=root \
  -e MYSQL_DATABASE=chatbot \
  mysql:8.0

# Initialize database
mysql -u root -p < docker/mysql/init.sql
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env:
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.2:1b
```

### 5. Start the Backend

```bash
make dev
# API at http://localhost:8000
```

### 6. Start the Frontend

```bash
make frontend
# Opens at http://localhost:3000
```

---

## Vision Models

For image analysis, pull and use vision-capable models:

```bash
# Pull vision models
podman exec rag-ollama ollama pull llava:7b
# or: ollama pull qwen2-vl:7b

# The application automatically detects Ollama and formats
# image requests correctly (images as base64 list)
```

### Sending Images via API

```bash
# Get auth token
TOKEN=$(curl -s http://localhost:8000/auth/login -X POST \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')

# Send image for analysis
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is in this image?",
    "images": [{"data": "BASE64_IMAGE_DATA", "media_type": "image/png"}],
    "stream": false
  }'
```

---

## Environment Configuration

Key settings in `.env`:

```bash
# LLM Provider (ollama, lmstudio, openai, custom)
LLM__PROVIDER=lmstudio
LLM__LMSTUDIO__BASE_URL=http://host.containers.internal:1234/v1
LLM__LMSTUDIO__MODEL=qwen/qwen3-vl-30b

# Alternative: Ollama
# LLM__PROVIDER=ollama
# LLM__OLLAMA__BASE_URL=http://ollama:11434
# LLM__OLLAMA__MODEL=mistral:latest

# Embeddings
EMBEDDING_MODEL=nomic-embed-text

# Database
MYSQL_HOST=mysql
MYSQL_PORT=3306
MYSQL_DATABASE=chatbot
MYSQL_USER=chatbot
MYSQL_PASSWORD=chatbot_password

# Redis (for embedding cache)
REDIS_URL=redis://redis:6379

# ChromaDB
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000

# Logging
LOGGING_LEVEL=INFO

# RAG Quality Settings (NEW)
RAG__BM25_ENABLED=true
RAG__BM25_WEIGHT=0.4
RAG__VECTOR_WEIGHT=0.6
RAG__KNOWLEDGE_GRAPH_ENABLED=true
RAG__KG_EXPANSION_DEPTH=2
RAG__VERIFICATION_ENABLED=true
RAG__CITATION_REQUIRED=true
RAG__MIN_CITATION_COVERAGE=0.7
```

---

## Monitoring

Open http://localhost:8000/admin.html and login with:
- **Username**: admin
- **Password**: admin123

The admin dashboard provides:
- Audit log viewer
- User management
- RAG evaluation runs
- System statistics

### 10. Upload Documents (Optional)

**Via UI:** Drag & drop files into the upload area

**Via CLI:**
```bash
# Place files in data/raw/
make ingest
```

**Via API:**
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "files=@document.pdf" \
  -F "client_name=John Doe" \
  -F "use_ocr=true"
```

---

## Docker Deployment

```bash
# Start everything with Docker
make docker-up

# Stop services
make docker-down
```

---

## API Endpoints

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
| `/models` | GET | List available models |
| `/models/switch` | POST | Switch LLM provider |
| `/health` | GET | Health check |
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

---

## Switching Providers at Runtime

```bash
# Switch to Ollama
curl -X POST http://localhost:8000/models/switch \
  -H "Content-Type: application/json" \
  -d '{"provider": "ollama", "model": "llama3"}'

# Switch to OpenAI
curl -X POST http://localhost:8000/models/switch \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-..."}'
```

---

## Configuration Reference

### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM__PROVIDER` | `lmstudio` | LLM backend: lmstudio, ollama, openai |
| `LLM__LMSTUDIO__BASE_URL` | `http://host.containers.internal:1234/v1` | LMStudio API endpoint |
| `LLM__LMSTUDIO__MODEL` | `qwen/qwen3-vl-30b` | LM Studio model |
| `LLM__OLLAMA__BASE_URL` | `http://ollama:11434` | Ollama API endpoint |
| `LLM__OLLAMA__MODEL` | `mistral:latest` | Ollama model |
| `LLM__TEMPERATURE` | `0.25` | Response creativity (0-1) |
| `LLM__MAX_TOKENS` | `1024` | Max response length |

### RAG Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model for embeddings |
| `VECTOR_STORE_PROVIDER` | `chromadb` | Vector DB: chromadb |
| `CHROMA_DB_PATH` | `./data/chroma` | ChromaDB data directory |
| `RAG__RERANKER_ENABLED` | `true` | Enable cross-encoder reranking |
| `RAG__INITIAL_FETCH_K` | `30` | Candidates before reranking |
| `RAG__RERANK_TOP_K` | `5` | Results after reranking |

### Hybrid Search & Knowledge Graph (NEW)

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG__BM25_ENABLED` | `true` | Enable BM25 keyword search |
| `RAG__BM25_WEIGHT` | `0.4` | BM25 weight in RRF fusion |
| `RAG__VECTOR_WEIGHT` | `0.6` | Vector weight in RRF fusion |
| `RAG__KNOWLEDGE_GRAPH_ENABLED` | `true` | Enable per-client knowledge graphs |
| `RAG__KG_EXPANSION_DEPTH` | `2` | Graph traversal depth |
| `RAG__KG_PERSIST_PATH` | `./data/knowledge_graphs` | KG storage path |

### Answer Verification (NEW)

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG__VERIFICATION_ENABLED` | `true` | Enable answer verification |
| `RAG__CITATION_REQUIRED` | `true` | Require source citations |
| `RAG__MIN_CITATION_COVERAGE` | `0.7` | Min 70% claims cited |
| `RAG__MIN_CONFIDENCE_THRESHOLD` | `0.3` | Evidence confidence threshold |

### Database & Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_HOST` | `localhost` | MySQL host |
| `DATABASE_PORT` | `3306` | MySQL port |
| `DATABASE_NAME` | `audit_logs` | MySQL database name |
| `DATABASE_USER` | `root` | MySQL username |
| `DATABASE_PASSWORD` | - | MySQL password (required) |
| `REDIS_URL` | `redis://redis:6379` | Redis URL |
| `JWT_SECRET_KEY` | - | JWT signing key (required) |
| `LOGGING_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `EVALUATION_ENABLED` | `true` | Enable scheduled evaluations |
| `EVALUATION_SCHEDULE_CRON` | `0 2 * * *` | Evaluation cron (UTC) |

---

## Troubleshooting

### LM Studio Connection Failed
- Ensure LM Studio server is running (Local Server → Start)
- Check the model is loaded (e.g., `qwen/qwen3-vl-30b`)
- Verify port 1234 is not blocked
- **For Docker/Podman:** Use `http://host.containers.internal:1234/v1` as the base URL
- Check `.env` has correct settings:
  ```bash
  LLM__PROVIDER=lmstudio
  LLM__LMSTUDIO__BASE_URL=http://host.containers.internal:1234/v1
  LLM__LMSTUDIO__MODEL=qwen/qwen3-vl-30b
  ```

### Embedding Errors
- Make sure `nomic-embed-text-v1.5` is loaded in LMStudio
- Or use Ollama: `ollama pull nomic-embed-text`

### ChromaDB Issues
- See [ChromaDB Setup Guide](./CHROMADB.md) for detailed troubleshooting
- Check data directory permissions: `chmod -R 755 ./data/chroma`

### MySQL Connection Failed
- Ensure MySQL is running: `systemctl status mysql` or `brew services list`
- Check credentials in `.env` match your MySQL setup
- Verify database exists: `mysql -u root -p -e "SHOW DATABASES;"`
- Re-run init script if needed: `mysql -u root -p < docker/mysql/init.sql`

### JWT Authentication Errors
- Ensure `JWT_SECRET_KEY` is set in `.env`
- Generate a new key: `openssl rand -hex 32`
- Clear browser local storage if tokens are stale

### Monitoring Stack Issues
- Check containers are running: `podman ps` or `docker ps`
- Verify Prometheus scraping: http://localhost:9090/targets
- Check Loki health: `curl http://localhost:3100/ready`
- See [Monitoring Guide](./MONITORING.md) for troubleshooting

---

## Deploying to Another Machine

### 1. Clone and Install

```bash
git clone https://github.com/GauravKumarYadav/rag-ai.git
cd rag-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Set Up Services

```bash
# MySQL
cd docker/mysql && podman-compose up -d && cd ../..

# ChromaDB (if using server mode)
cd docker/chromadb && podman-compose up -d && cd ../..

# Monitoring (optional)
cd docker/monitoring && podman-compose up -d && cd ../..
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings:
# - DATABASE_PASSWORD
# - JWT_SECRET_KEY
# - LLM provider settings
```

### 4. Start Application

```bash
make dev        # Backend
make frontend   # Frontend (new terminal)
```

### 5. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Get JWT token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')

# Test authenticated endpoint
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/auth/me
```

---

## Additional Documentation

- [README.md](../README.md) - Project overview
- [Architecture Guide](./ARCHITECTURE.md) - System architecture
- [ChromaDB Setup](./CHROMADB.md) - Vector database configuration
- [Monitoring Guide](./MONITORING.md) - Prometheus, Loki, Grafana
- [Evaluation Guide](./EVALUATION.md) - RAG evaluation framework
- [API Docs](http://localhost:8000/docs) - Interactive API documentation (when running)
