# Development Guide

## Prereqs
- Python 3.11+
- Node not required (frontend is static HTML)
- LM Studio running locally (or any OpenAI-compatible endpoint)

## Backend (local, no containers)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

export LLM__PROVIDER=lmstudio
export LLM__LMSTUDIO__BASE_URL=http://127.0.0.1:1234/v1
export LLM__LMSTUDIO__MODEL=<your-model>

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Containers (recommended)
```bash
podman-compose -f docker-compose.microservices.yml up -d
# or docker-compose ...
```
Services: chat-api, chromadb, redis, nginx (serves frontend at http://localhost).

## Frontend
- Served by nginx from `frontend/`. For local preview without nginx, open `frontend/index.html` in the browser or serve via any static server.

## Testing (selective)
```bash
pytest tests/test_agents.py
pytest tests/test_api_endpoints.py
```

## Common dev workflows
- Logs: `podman-compose logs -f chat-api`
- Health: `curl http://localhost/health`
- Auth token:
  ```bash
  TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')
  ```
- Upload & ask: see [API.md](./API.md)

## Notes
- Upload pipeline is synchronous; documents are available immediately after the upload response.
- Always pass `client_id` to `/chat` and `/documents/upload`; the agent uses only that client’s data plus global.
