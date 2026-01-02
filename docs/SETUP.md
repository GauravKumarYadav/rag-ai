# Local Multimodal AI Chatbot

A fully local AI chatbot with RAG, long-term memory, vision support, and multi-provider LLM backend.

## Features
- 🤖 **Multi-Provider LLM**: LMStudio, Ollama, OpenAI, or any OpenAI-compatible endpoint
- 📚 **RAG**: Upload documents and query them with semantic search
- 🧠 **Long-term Memory**: Automatic conversation summarization and recall
- 🖼️ **Vision Support**: Send images for analysis (with vision models)
- 🔄 **Real-time Streaming**: WebSocket and SSE streaming responses
- 🎨 **Modern Frontend**: Clean chat interface with settings panel

## Quick Start

### 1. Install Dependencies
```bash
cd /Users/g0y01hx/Desktop/personal_work/chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Set Up LLM Provider

#### Option A: LMStudio (Recommended for local)
1. Download [LMStudio](https://lmstudio.ai)
2. Download a model (e.g., `Qwen2.5-VL`, `Llama-3`, `Mistral`)
3. Go to "Local Server" tab → Load model → Start Server
4. Server runs at `http://localhost:1234`

#### Option B: Ollama
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3
ollama serve  # Runs at http://localhost:11434
```

#### Option C: OpenAI
Set your API key in `.env`:
```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env to match your provider settings
```

### 4. Start the Backend
```bash
make dev
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 5. Start the Frontend
```bash
# In a new terminal
make frontend
# Opens at http://localhost:3000
```

### 6. Upload Documents (Optional)
```bash
# Place files in data/raw/
make ingest
```

## Docker Deployment
```bash
# Start everything with Docker
make docker-up

# Stop services
make docker-down
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /chat` | Chat with streaming support |
| `WS /chat/ws/{id}` | WebSocket chat |
| `POST /documents/upload` | Upload documents |
| `POST /documents/search` | Search documents |
| `GET /models` | List available models |
| `POST /models/switch` | Switch LLM provider |
| `GET /conversations` | List conversations |
| `GET /health` | Health check |

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

## Configuration

See `.env.example` for all options:
- `LLM_PROVIDER`: lmstudio, ollama, openai, custom
- `LLM_TEMPERATURE`: Response creativity (0-1)
- `LLM_MAX_TOKENS`: Max response length
- `EMBEDDING_MODEL`: Model for document embeddings
- Default model name: match `LMSTUDIO_MODEL`
- Enable streaming and image uploads.

## 6) Notes
- The backend exposes:
  - `GET /health` – checks LM Studio reachability.
  - `POST /chat` – chat with text/images, streams Server-Sent Events.
- Long-term memory summaries are stored in the `memories` Chroma collection.
- Everything runs locally; no cloud calls are required.

