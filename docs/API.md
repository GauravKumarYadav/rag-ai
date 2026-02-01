# API Reference (Essential Endpoints)

All endpoints are served via nginx on `http://localhost`. Authenticate first to obtain `Bearer` token.

## Auth
| Method | Path | Body | Notes |
| --- | --- | --- | --- |
| POST | `/auth/login` | `{"username","password"}` | Returns `access_token` |
| GET | `/auth/me` | — | Requires `Authorization: Bearer <token>` |

## Clients
| Method | Path | Body | Notes |
| --- | --- | --- | --- |
| GET | `/clients` | — | List clients |
| POST | `/clients` | `{"name": "...", "aliases": []}` | Create client |
| GET | `/clients/{client_id}` | — | Client details |

## Documents
| Method | Path | Body | Notes |
| --- | --- | --- | --- |
| POST | `/documents/upload` | multipart form: `files=@...`, `client_id=<id>` | Upload; synchronous extract + embed |
| POST | `/documents/search` | `{"query": "...", "client_id": "<id>"}` | Semantic search |
| GET | `/documents/stats` | `?client_id=<id>` | Basic stats |

## Chat
| Method | Path | Body | Notes |
| --- | --- | --- | --- |
| POST | `/chat` | `{"message": "...", "client_id": "<id>", "conversation_id": "<id>", "stream": false}` | Main RAG chat. Pass `client_id` every time. |

### Chat Response Fields
- `response`: model reply (with citations if present)
- `sources`: list of retrieval hits with metadata

## Health
| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | Basic service health |

## Curl Examples
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')

# Upload
curl -s -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "files=@/path/to/doc.pdf" \
  -F "client_id=<client-id>"

# Ask
curl -s -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"what does my document say?","client_id":"<client-id>","stream":false}'
```

## Notes
- Always set `client_id`; the agent uses only that client’s collection plus global.
- Upload is synchronous; data is available immediately after the upload response.
