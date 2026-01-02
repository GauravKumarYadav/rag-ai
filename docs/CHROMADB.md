# ChromaDB Setup Guide

This guide covers installing and running ChromaDB as both an embedded database and as a standalone service.

## Overview

ChromaDB is the default vector database for this application. It can run in two modes:

| Mode | Use Case | Configuration |
|------|----------|---------------|
| **Embedded** (default) | Development, single-machine | No extra setup needed |
| **Server Mode** | Production, multi-service | Run ChromaDB as a service |

---

## Option 1: Embedded Mode (Default)

This is the simplest setup - ChromaDB runs inside the Python application.

### Requirements
- Python 3.11+
- `chromadb` package (included in requirements.txt)

### Setup
```bash
# Already installed with the application
pip install chromadb

# Data is stored locally in:
# ./data/chroma/
```

### Configuration
```bash
# .env
VECTOR_STORE_PROVIDER=chromadb
CHROMA_DB_PATH=./data/chroma
```

**Pros:** Simple, no extra services to manage  
**Cons:** Single process access, not suitable for distributed systems

---

## Option 2: ChromaDB Server Mode

Run ChromaDB as a standalone HTTP service for production deployments.

### Method A: Using Docker (Recommended)

```bash
# Pull the official ChromaDB image
docker pull chromadb/chroma:latest

# Run ChromaDB server
docker run -d \
  --name chromadb \
  -p 8001:8000 \
  -v $(pwd)/data/chroma:/chroma/chroma \
  -e ANONYMIZED_TELEMETRY=FALSE \
  -e ALLOW_RESET=TRUE \
  chromadb/chroma:latest

# Verify it's running
curl http://localhost:8001/api/v2/heartbeat
# Response: {"nanosecond heartbeat": ...}
```

### Method B: Using Docker Compose

Create `docker/chromadb/docker-compose.yml`:

```yaml
version: '3.8'

services:
  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb
    ports:
      - "8001:8000"
    volumes:
      - ../../data/chroma:/chroma/chroma
    environment:
      - ANONYMIZED_TELEMETRY=FALSE
      - ALLOW_RESET=TRUE
      - CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token_authn.TokenAuthenticationServerProvider
      - CHROMA_SERVER_AUTH_TOKEN_TRANSPORT_HEADER=Authorization
      # Uncomment to enable authentication:
      # - CHROMA_SERVER_AUTH_CREDENTIALS=your-secret-token
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v2/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  chroma_data:
```

Run with:
```bash
cd docker/chromadb
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Method C: Native Installation (Without Docker)

```bash
# Install ChromaDB with server dependencies
pip install chromadb[server]

# Run the server
chroma run --host 0.0.0.0 --port 8001 --path ./data/chroma

# Or run in background
nohup chroma run --host 0.0.0.0 --port 8001 --path ./data/chroma > chroma.log 2>&1 &
```

### Method D: systemd Service (Linux Production)

Create `/etc/systemd/system/chromadb.service`:

```ini
[Unit]
Description=ChromaDB Vector Database
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/chromadb
ExecStart=/opt/chromadb/venv/bin/chroma run --host 0.0.0.0 --port 8001 --path /var/lib/chromadb
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/chromadb

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable chromadb
sudo systemctl start chromadb
sudo systemctl status chromadb
```

### Method E: macOS launchd Service

Create `~/Library/LaunchAgents/com.chromadb.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.chromadb</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USER/.venv/bin/chroma</string>
        <string>run</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8001</string>
        <string>--path</string>
        <string>/Users/YOUR_USER/data/chroma</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/chromadb.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/chromadb.error.log</string>
</dict>
</plist>
```

Load the service:
```bash
launchctl load ~/Library/LaunchAgents/com.chromadb.plist
launchctl start com.chromadb

# Check status
launchctl list | grep chromadb

# View logs
tail -f /tmp/chromadb.log
```

---

## Configuring the Application for Server Mode

Update your `.env` to connect to the ChromaDB server:

```bash
# Vector Store Configuration
VECTOR_STORE_PROVIDER=chromadb
VECTOR_STORE_URL=http://localhost:8001  # ChromaDB server URL

# If using authentication
# VECTOR_STORE_API_KEY=your-secret-token
```

Update `backend/app/config.py` if needed - the modular architecture already supports this.

---

## Verifying the Setup

### Check ChromaDB Health

```bash
# For embedded mode
python -c "import chromadb; c = chromadb.Client(); print('OK')"

# For server mode
curl http://localhost:8001/api/v2/heartbeat
```

### Test from the Application

```bash
cd backend
python -c "
from app.rag import get_vector_store
store = get_vector_store()
print(f'Provider: {store.get_stats()[\"provider\"]}')
print(f'Health: {store.health_check()}')
print(f'Documents: {store.get_stats()[\"document_count\"]}')
"
```

---

## Troubleshooting

### Port Already in Use
```bash
# Find what's using port 8001
lsof -i :8001

# Kill the process
kill -9 <PID>
```

### Permission Denied on Data Directory
```bash
# Fix permissions
sudo chown -R $USER:$USER ./data/chroma
chmod -R 755 ./data/chroma
```

### Docker Container Won't Start
```bash
# Check logs
docker logs chromadb

# Common fix: remove old container
docker rm -f chromadb
docker run ... # (run command again)
```

### Connection Refused
```bash
# Verify ChromaDB is running
docker ps | grep chroma
# or
ps aux | grep chroma

# Check firewall (Linux)
sudo ufw allow 8001/tcp
```

### Data Not Persisting
Ensure the volume mount is correct:
```bash
# Docker: check volume
docker inspect chromadb | grep -A 5 "Mounts"

# Should show your local path mapped to /chroma/chroma
```

---

## Performance Tuning

### For Large Datasets (>100k documents)

```bash
# Increase memory for Docker
docker run -d \
  --name chromadb \
  --memory=4g \
  --cpus=2 \
  -p 8001:8000 \
  -v $(pwd)/data/chroma:/chroma/chroma \
  chromadb/chroma:latest
```

### Enable Persistent WAL

```python
# In your application, use persistent client
import chromadb
client = chromadb.PersistentClient(
    path="./data/chroma",
    settings=chromadb.Settings(
        anonymized_telemetry=False,
        allow_reset=True,
    )
)
```

---

## Migration: Embedded → Server Mode

If you have existing data in embedded mode and want to switch to server mode:

1. **Stop the application**
2. **Start ChromaDB server** pointing to the same data directory:
   ```bash
   docker run -d \
     -p 8001:8000 \
     -v $(pwd)/data/chroma:/chroma/chroma \
     chromadb/chroma:latest
   ```
3. **Update `.env`**:
   ```bash
   VECTOR_STORE_URL=http://localhost:8001
   ```
4. **Restart the application**

Your existing data will be available through the server.

---

## Security Considerations

### Enable Authentication (Production)

```bash
# In docker-compose or docker run
-e CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token_authn.TokenAuthenticationServerProvider
-e CHROMA_SERVER_AUTH_CREDENTIALS=your-secure-token-here
```

Then in `.env`:
```bash
VECTOR_STORE_API_KEY=your-secure-token-here
```

### Network Security

- Run ChromaDB on internal network only (not exposed to internet)
- Use a reverse proxy (nginx) with TLS for external access
- Consider VPN for remote access

---

## Next Steps

- [README.md](../README.md) - Full application setup
- [SETUP.md](./SETUP.md) - Quick start guide
- [Switching Vector Store Providers](../backend/app/rag/factory.py) - Use Pinecone, Weaviate, etc.
