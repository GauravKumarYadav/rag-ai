# 🤖 RAG AI Chatbot

A fully local, privacy-focused AI chatbot with RAG (Retrieval-Augmented Generation), long-term memory, vision support, and multi-client data isolation.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.4-orange)

## ✨ Features

- 🔒 **100% Local & Private** - All data stays on your machine
- 🤖 **Multi-Provider LLM** - LMStudio, Ollama, OpenAI, or any OpenAI-compatible API
- 📚 **RAG Pipeline** - Upload documents (PDF, DOCX, TXT, images) with OCR support
- 🧠 **Long-term Memory** - Automatic conversation summarization and recall
- 👥 **Multi-Client Isolation** - Separate data per client with auto-detection
- 🖼️ **Vision Support** - Analyze images with vision-capable models
- 🔄 **Real-time Streaming** - SSE and WebSocket streaming responses
- 📊 **Markdown Tables** - Properly formatted tables in responses
- 💾 **Chat History** - Save, load, and export conversations

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **LMStudio** (recommended) or Ollama for local LLM inference
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/GauravKumarYadav/rag-ai.git
cd rag-ai
```

### 2. Install Dependencies

```bash
# Create virtual environment and install packages
make install

# Or manually:
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 3. Set Up LLM Provider

#### Option A: LMStudio (Recommended)

1. Download [LMStudio](https://lmstudio.ai)
2. Download models:
   - **Chat Model**: `Qwen2.5-VL`, `Llama-3`, `Mistral`, etc.
   - **Embedding Model**: `nomic-embed-text-v1.5` (required for RAG)
3. Go to **Local Server** tab → Load models → **Start Server**
4. Server runs at `http://localhost:1234`

#### Option B: Ollama

```bash
# Install from https://ollama.ai
ollama pull llama3
ollama pull nomic-embed-text
ollama serve  # Runs at http://localhost:11434
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work for LMStudio)
```

### 5. Start the Application

**Terminal 1 - Backend:**
```bash
source .venv/bin/activate
make dev
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Terminal 2 - Frontend:**
```bash
make frontend
# Opens at http://localhost:3000
```

### 6. Upload Documents (Optional)

**Via UI:**
- Drag & drop files into the upload area
- Enter client name when prompted (auto-creates client)

**Via CLI:**
```bash
# Place files in data/raw/ then:
make ingest
```

**Via API:**
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "files=@document.pdf" \
  -F "client_name=John Doe" \
  -F "use_ocr=true"
```

## 📁 Project Structure

```
rag-ai/
├── backend/
│   ├── app/
│   │   ├── api/routes/      # API endpoints
│   │   ├── clients/         # LLM client adapters
│   │   ├── memory/          # Session & long-term memory
│   │   ├── models/          # Pydantic schemas
│   │   ├── rag/             # Vector store & retrieval
│   │   ├── services/        # Business logic
│   │   ├── config.py        # Settings
│   │   └── main.py          # FastAPI app
│   └── requirements.txt
├── frontend/
│   └── index.html           # Single-page chat UI
├── data/
│   ├── chroma/              # Vector database
│   └── raw/                 # Documents to ingest
├── docker/                  # Docker configs
├── tests/                   # Unit tests
├── Makefile                 # Dev commands
└── README.md
```

## 🔧 Configuration

Edit `.env` or `backend/app/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `lmstudio` | `lmstudio`, `ollama`, `openai` |
| `LLM_API_URL` | `http://localhost:1234/v1` | LLM API endpoint |
| `LLM_MODEL` | `qwen2.5-vl-7b-instruct` | Model name |
| `EMBEDDING_MODEL` | `text-embedding-nomic-embed-text-v1.5` | Embedding model |
| `CHROMA_DB_PATH` | `data/chroma` | Vector DB location |

## 📡 API Endpoints

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

### LMStudio Connection Failed
- Ensure LMStudio server is running (Local Server → Start)
- Check the model is loaded
- Verify port 1234 is not blocked

### Embedding Errors
- Make sure you have `nomic-embed-text-v1.5` loaded in LMStudio
- Or switch to Ollama: `ollama pull nomic-embed-text`

### OCR Not Working
- Docling requires additional dependencies for some formats
- Try `pip install docling[all]` for full support

### Tables Not Rendering
- Refresh the frontend page to load latest JavaScript
- Ensure streaming is enabled in settings

## 📄 License

MIT License - Feel free to use and modify.

## 🙏 Acknowledgments

- [LMStudio](https://lmstudio.ai) - Local LLM inference
- [ChromaDB](https://www.trychroma.com) - Vector database
- [FastAPI](https://fastapi.tiangolo.com) - Backend framework
- [Docling](https://github.com/DS4SD/docling) - Document parsing with OCR
