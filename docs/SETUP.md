# Setup Guide (Current Stack)

Lightweight, single compose stack (chat API + ChromaDB + Redis + Nginx + frontend). LM Studio is the default LLM provider.

## Prerequisites
- Podman or Docker
- jq (for CLI testing)
- LM Studio running locally (or any OpenAI-compatible endpoint)

## Quick Start (Containers)
```bash
git clone <repo-url>
cd chatbot
podman-compose -f docker-compose.microservices.yml up -d   # or: docker-compose ...
```
Services:
- chat-api (FastAPI, port 8000 internal via nginx)
- chromadb (vector store)
- redis (session/memory)
- nginx (serves frontend at http://localhost)

## Configure LLM (LM Studio)
1) Start LM Studio server (Local Server → start)  
2) Load a model (e.g., qwen/qwen3-vl-30b or any text-capable model)  
3) Ensure base URL is reachable from containers: `http://host.containers.internal:1234/v1`

Set in `.env`:
```bash
LLM__PROVIDER=lmstudio
LLM__LMSTUDIO__BASE_URL=http://host.containers.internal:1234/v1
LLM__LMSTUDIO__MODEL=<your-model-name>
LLM__TEMPERATURE=0.2
LLM__MAX_TOKENS=1024
```

## Run
```bash
podman-compose -f docker-compose.microservices.yml up -d
```
- Frontend: http://localhost  
- API health: `curl http://localhost/health`

## Auth (default dev)
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')
```

## Upload & Query
```bash
# Upload
curl -s -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "files=@mydoc.pdf" \
  -F "client_id=<client-id>"

# Ask about it
curl -s -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"what does my document say?","client_id":"<client-id>","stream":false}'
```

## Local Development (without containers)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
export LLM__PROVIDER=lmstudio
export LLM__LMSTUDIO__BASE_URL=http://127.0.0.1:1234/v1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Serve `frontend/index.html` via any static server (or use the bundled nginx from compose).

## Notes
- No MySQL, no monitoring stack, no background workers. Upload is synchronous: text extraction → chunking → embeddings → Chroma. Data is available immediately after the upload response.
- Client isolation: pass `client_id` on every `/chat` and `/documents/upload` request. The agent uses only that client's collection plus global. No query-based client detection.

## Data Persistence
Data directories use bind mounts and persist across container restarts:
- ChromaDB: `./data/chroma`
- Redis: `./data/redis`
- BM25 index: `./data/bm25`
- Knowledge graphs: `./data/knowledge_graphs`

> Running `podman-compose down -v` will NOT delete your data (bind mounts are not affected by the -v flag).

## Troubleshooting Quick Checks
- LM Studio unreachable from containers: use `host.containers.internal` instead of `127.0.0.1`
- Empty answers: ensure the correct `client_id` is set and documents are uploaded; intent is now LLM-classified (no hardcoded patterns)
- Data paths: `./data/chroma`, `./data/redis`, `./data/bm25`, `./data/knowledge_graphs` (bind mounts in compose)
