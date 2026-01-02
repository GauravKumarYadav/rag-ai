# Local Multimodal AI Chatbot - Setup Guide

A fully local AI chatbot with RAG, long-term memory, vision support, and multi-provider LLM backend.

## Features

- 🤖 **Multi-Provider LLM**: LMStudio, Ollama, OpenAI, or any OpenAI-compatible endpoint
- 📚 **RAG**: Upload documents and query them with semantic search
- 🧠 **Long-term Memory**: Automatic conversation summarization and recall
- 🖼️ **Vision Support**: Send images for analysis (with vision models)
- 🔄 **Real-time Streaming**: WebSocket and SSE streaming responses
- 🎨 **Modern Frontend**: Clean chat interface with settings panel
- 🔌 **Modular Vector Store**: ChromaDB (default), with support for Pinecone, Weaviate, etc.

## Prerequisites

- **Python 3.11+**
- **Git**
- **LMStudio** or **Ollama** (for local LLM inference)

---

## Quick Start

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/GauravKumarYadav/rag-ai.git
cd rag-ai

# Create virtual environment and install packages
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Set Up LLM Provider

#### Option A: LMStudio (Recommended)

1. Download [LMStudio](https://lmstudio.ai)
2. Download models:
   - **Chat Model**: `Qwen2.5-VL`, `Llama-3`, `Mistral`
   - **Embedding Model**: `nomic-embed-text-v1.5` (required for RAG)
3. Go to **Local Server** tab → Load models → **Start Server**
4. Server runs at `http://localhost:1234`

#### Option B: Ollama

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3
ollama pull nomic-embed-text  # For embeddings
ollama serve  # Runs at http://localhost:11434
```

#### Option C: OpenAI

```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...
```

### 3. Set Up ChromaDB (Vector Database)

ChromaDB runs automatically in **embedded mode** - no extra setup needed!

**For production (server mode):**

```bash
# Run ChromaDB as a Docker service
docker run -d \
  --name chromadb \
  -p 8001:8000 \
  -v $(pwd)/data/chroma:/chroma/chroma \
  chromadb/chroma:latest

# Verify it's running
curl http://localhost:8001/api/v2/heartbeat

# Update .env to use server mode
echo "VECTOR_STORE_URL=http://localhost:8001" >> .env
```

📖 See [ChromaDB Setup Guide](./CHROMADB.md) for detailed instructions including systemd/launchd services.

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env to match your provider settings
```

Key settings in `.env`:
```bash
# LLM Provider
LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=qwen2.5-vl-7b-instruct

# Embeddings
EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5

# Vector Store (modular architecture)
VECTOR_STORE_PROVIDER=chromadb
CHROMA_DB_PATH=./data/chroma
# VECTOR_STORE_URL=http://localhost:8001  # Uncomment for server mode
```

### 5. Start the Backend

```bash
source .venv/bin/activate
make dev
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 6. Start the Frontend

```bash
# In a new terminal
make frontend
# Opens at http://localhost:3000
```

### 7. Upload Documents (Optional)

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

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `lmstudio` | LLM backend: lmstudio, ollama, openai |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LMStudio API endpoint |
| `EMBEDDING_MODEL` | `text-embedding-nomic-embed-text-v1.5` | Model for embeddings |
| `VECTOR_STORE_PROVIDER` | `chromadb` | Vector DB: chromadb, pinecone, etc. |
| `CHROMA_DB_PATH` | `./data/chroma` | ChromaDB data directory |
| `VECTOR_STORE_URL` | (none) | Remote ChromaDB server URL |
| `LLM_TEMPERATURE` | `0.25` | Response creativity (0-1) |
| `LLM_MAX_TOKENS` | `1024` | Max response length |

---

## Troubleshooting

### LMStudio Connection Failed
- Ensure LMStudio server is running (Local Server → Start)
- Check the model is loaded
- Verify port 1234 is not blocked

### Embedding Errors
- Make sure `nomic-embed-text-v1.5` is loaded in LMStudio
- Or use Ollama: `ollama pull nomic-embed-text`

### ChromaDB Issues
- See [ChromaDB Setup Guide](./CHROMADB.md) for detailed troubleshooting
- Check data directory permissions: `chmod -R 755 ./data/chroma`

---

## Additional Documentation

- [README.md](../README.md) - Project overview
- [ChromaDB Setup](./CHROMADB.md) - Vector database configuration
- [API Docs](http://localhost:8000/docs) - Interactive API documentation (when running)
