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
- Data path: `./data/chroma` (mounted in compose). Ensure writable: `chmod -R 755 data/chroma`.
- Health: check container logs: `podman-compose logs -f chromadb`.

## Redis / Memory
- Redis stores session/memory; if unavailable, conversations may not persist.
- Health: `podman-compose logs -f redis`.

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
