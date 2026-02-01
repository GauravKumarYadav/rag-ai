# Troubleshooting

## LM Studio / LLM Connectivity
- Ensure LM Studio server is running (Local Server → Start) and model is loaded.
- From containers use: `http://host.containers.internal:1234/v1`.
- Check `.env`: `LLM__PROVIDER=lmstudio`, `LLM__LMSTUDIO__BASE_URL=...`, `LLM__LMSTUDIO__MODEL=...`.
- Test: `curl http://host.containers.internal:1234/v1/models`.

## Empty Answers or Missing Documents
- Verify `client_id` is set on `/chat` and `/documents/upload`.
- Confirm documents exist for that client: upload again and query; data is available immediately after upload (synchronous pipeline).
- Intent is LLM-classified (no hardcoded patterns). If you ask “what documents do I have?” you’ll get a list; “what does my document say?” will retrieve content.

## ChromaDB Issues
- ChromaDB data is in named volume `rag-chroma-data` (persists across `compose down`).
- Health: check container logs: `podman compose logs -f chromadb`.

## Redis / Memory
- Redis data is in named volume `rag-redis-data` (persists across `compose down`).
- Redis stores session/memory; if unavailable, conversations may not persist.
- Health: `podman compose logs -f redis`.

## Data Persistence
ChromaDB, Redis, and BM25 use **named volumes** and survive `podman compose down`:
- ChromaDB: volume `rag-chroma-data`
- Redis: volume `rag-redis-data`
- BM25 index: volume `rag-bm25-data`
- Ingested files: bind mount `./data/raw` (create this folder on the host)

**Do not run `podman compose down -v`** unless you want to **delete all chunks, vectors, and session data**; `-v` removes these named volumes. To backup a volume: `podman run --rm -v rag-chroma-data:/data -v $(pwd):/backup alpine tar czf /backup/chroma-backup.tar.gz -C /data .`

## Auth Errors
- Ensure `JWT_SECRET_KEY` is set.
- Default dev creds: `admin` / `admin123` (can be changed in DB).

## Upload Failures
- Only Docling extraction is used (no fallbacks). If extraction fails, the upload will error.
- Large or scanned PDFs can be slow; the endpoint blocks until embedding completes.

## Nginx / Frontend
- Frontend is served at `http://localhost`. If blank, ensure nginx is running: `podman-compose ps`.

## Quick Health Checks
```bash
curl http://localhost/health
podman-compose ps
podman-compose logs -f chat-api
```

## Resource Baseline (CPU, Memory, Volumes)

To measure container CPU, memory, and volume usage (e.g. before/after optimization):

1. **Run the baseline script** from the repo root:
   ```bash
   chmod +x scripts/baseline_resources.sh
   ./scripts/baseline_resources.sh
   ```

2. **Manual commands** (if not using Docker):
   - Disk usage: `docker system df -v` (or `podman system df -v`)
   - Live container stats: `docker stats --no-stream` (or `podman stats --no-stream`)
   - Volume inspect: `docker volume inspect rag-chroma-data rag-bm25-data rag-redis-data`

3. **Interpretation**: Named volumes `rag-chroma-data`, `rag-bm25-data`, and `rag-redis-data` plus the `chat-api` image and build cache account for most disk. High CPU under load usually comes from the reranker and Docling running in the same process; see docs for concurrency and resource limits.

## Docker build cache and image size

To reduce disk used by Docker images and build cache:

1. **Prune build cache** (reclaim space; next build will be slower until cache repopulates):
   ```bash
   docker builder prune -f
   # or: podman buildah prune -f
   ```

2. **Prune unused images and containers**:
   ```bash
   docker system prune -a
   # or: podman system prune -a
   ```
   Use with care; this removes all unused images and stopped containers.

3. **chat-api image**: The backend uses a multi-stage Dockerfile so the final image does not include build tools (gcc, build-essential), which keeps the runtime image smaller.

## Pruning and retention (Chroma, Redis, BM25)

To cap volume growth or remove old/unused data (optional; does not change retrieval for active data):

1. **ChromaDB** (`rag-chroma-data`): Data lives in the Chroma container. To shrink:
   - Delete specific collections via the Chroma HTTP API or by reinitializing the store (see Chroma docs).
   - To wipe and start fresh: `docker compose down -v` removes named volumes (including Chroma); only do this if you are okay losing all vectors and documents.

2. **Redis** (`rag-redis-data`): Redis is already capped in memory (`--maxmemory 1gb` in docker-compose). On-disk AOF can be compacted by restarting Redis or running `BGREWRITEAOF` inside the container. To wipe Redis data: remove the volume or run `FLUSHDB`/`FLUSHALL` inside the container (conversations/summaries will be lost).

3. **BM25** (`rag-bm25-data`): Persisted as JSON files per client (`bm25_{client_id}.json`). To reclaim space:
   - Delete files for clients you no longer need: e.g. remove `bm25_old_client.json` from the volume mount path. The app will create a new empty index for that client on next use.
   - BM25 now persists **tokens + metadata only** (content is resolved from Chroma at search time), so existing files will shrink after the next persist cycle; new persists do not store full content.
