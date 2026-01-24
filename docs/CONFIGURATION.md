# Configuration Reference (Essentials)

All settings are read from `.env`. Key variables only; avoid unused legacy settings.

## LLM
| Variable | Example | Notes |
| --- | --- | --- |
| `LLM__PROVIDER` | `lmstudio` | Provider: `lmstudio` (default) |
| `LLM__LMSTUDIO__BASE_URL` | `http://host.containers.internal:1234/v1` | Use host.containers.internal from containers |
| `LLM__LMSTUDIO__MODEL` | `qwen/qwen3-vl-30b` | Any loaded model name |
| `LLM__TEMPERATURE` | `0.2` | 0-1 |
| `LLM__MAX_TOKENS` | `1024` | Max response length |

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
