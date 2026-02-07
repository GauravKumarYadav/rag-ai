# Configuration Reference (Essentials)

All settings are read from `.env`. Key variables only; avoid unused legacy settings.

## LLM
| Variable | Example | Notes |
| --- | --- | --- |
| `LLM__PROVIDER` | `lmstudio` | Providers: `lmstudio` (default), `ollama`, `groq`, `openai`, `custom` |
| `LLM__LMSTUDIO__BASE_URL` | `http://host.containers.internal:1234/v1` | LM Studio OpenAI-compatible endpoint |
| `LLM__LMSTUDIO__MODEL` | `qwen/qwen3-vl-30b` | Any loaded model name |
| `LLM__OLLAMA__BASE_URL` | `https://3bf866fda35d.ngrok-free.app` | Remote Ollama host (no `/api` suffix) |
| `LLM__OLLAMA__MODEL` | `llama3` | Must exist on that Ollama instance |
| `LLM__GROQ__BASE_URL` | `https://api.groq.com/openai/v1` | Groq OpenAI-compatible endpoint |
| `LLM__GROQ__MODEL` | `llama-3.3-70b-versatile` | Default Groq model (free tier) |
| `LLM__GROQ__API_KEY` | *(secret)* | Required; create at https://console.groq.com/keys |
| `LLM__TEMPERATURE` | `0.2` | 0-1 |
| `LLM__MAX_TOKENS` | `1024` | Max response length |

Embeddings:
- `EMBEDDING_PROVIDER=auto` (follows `LLM__PROVIDER`; override to force a provider)
- `EMBEDDING_MODEL` must exist on the chosen provider (e.g., `text-embedding-nomic-embed-text-v1.5` on LM Studio or an Ollama embedding model like `nomic-embed-text`).

Remote Ollama (ngrok):
- Run Ollama on the other laptop (`ollama serve`).
- Expose the Ollama port via ngrok: `ngrok http 11434`.
- Set `LLM__PROVIDER=ollama` and paste the ngrok HTTPS URL into `LLM__OLLAMA__BASE_URL`.
- No auth headers are required unless ngrok was configured otherwise.

## RAG & Storage
| Variable | Example | Notes |
| --- | --- | --- |
| `RAG__CHROMA_DB_PATH` | `./data/chroma` | Local Chroma storage |
| `RAG__URL` | *(empty)* | Leave empty for local persistent client |
| `RAG__COLLECTION_PREFIX` | *(empty)* | Optional prefix |
| `REDIS_URL` | `redis://redis:6379` | Session/memory store |

## Server
| Variable | Example | Notes |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | API bind host |
| `PORT` | `8000` | API port (internal; proxied by nginx) |
| `LOG_LEVEL` | `info` | Logging level |

## Auth
| Variable | Example | Notes |
| --- | --- | --- |
| `JWT_SECRET_KEY` | `<hex>` | Required |
| `JWT_EXPIRE_MINUTES` | `1440` | Token TTL |

## Client Context
- Always pass `client_id` on `/chat` and `/documents/upload`. The agent uses only that client’s collection plus global.
- No client inference from text; it relies solely on the provided `client_id`.

## Notes
- Upload is synchronous: extract → chunk → embed → store. Data is available immediately after the upload response.
- No MySQL, no monitoring stack, no background workers in this stack.
