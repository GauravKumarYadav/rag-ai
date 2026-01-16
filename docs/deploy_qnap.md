# Deploying RAG AI Chatbot on QNAP NAS

This guide covers deploying the RAG AI Chatbot on a QNAP NAS using Container Station.

## Prerequisites

### QNAP NAS Requirements

- Intel/AMD x86_64 CPU (ARM-based NAS may have limited model compatibility)
- Minimum 8GB RAM (16GB+ recommended for LLM inference)
- Container Station installed from QNAP App Center
- SSH access enabled (optional but recommended)

### Install Container Station

1. Go to **App Center** → Search "Container Station" → Install
2. Open Container Station and complete initial setup

---

## Step 1: Transfer Project Files to QNAP

### Option A: Via SSH/SCP

```bash
# From your local machine
scp -r /path/to/chatbot admin@<QNAP_IP>:/share/Container/chatbot
```

### Option B: Via File Station

1. Open **File Station** on QNAP web interface
2. Navigate to `/Container/` or create a new folder
3. Upload the entire `chatbot` project folder

### Option C: Via Git (Recommended)

```bash
# SSH into QNAP
ssh admin@<QNAP_IP>

# Navigate to Container share
cd /share/Container

# Clone the repository
git clone <your-repo-url> chatbot
cd chatbot
```

---

## Step 2: Create Environment Configuration

SSH into your QNAP or use the Container Station terminal:

```bash
cd /share/Container/chatbot

# Create .env file
cat > .env << 'EOF'
# LLM Provider (choose one)
LLM__PROVIDER=lmstudio
LLM__LMSTUDIO__BASE_URL=http://<YOUR_PC_IP>:1234/v1
LLM__LMSTUDIO__MODEL=qwen/qwen3-vl-30b

# Or use Ollama (included in stack)
# LLM__PROVIDER=ollama
# LLM__OLLAMA__MODEL=llama3.2:1b

# Embeddings
EMBEDDING_PROVIDER=lmstudio
EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5

# Database
MYSQL_ROOT_PASSWORD=YourSecurePassword123!
MYSQL_DATABASE=audit_logs

# JWT Security (generate with: openssl rand -hex 32)
JWT__SECRET_KEY=your-secure-random-key-here

# LLM Settings
LLM__TEMPERATURE=0.35
LLM__MAX_TOKENS=4096
EOF
```

---

## Step 3: Create Data Directories

```bash
cd /share/Container/chatbot

# Create required directories with proper permissions
mkdir -p data/chroma-docker data/bm25 data/knowledge_graphs data/raw
chmod -R 755 data/
```

---

## Step 4: Deploy Using Container Station

### Method A: Using Container Station Web UI

1. Open **Container Station** → **Create**
2. Select **Create Application**
3. Choose **Docker Compose**
4. Set Application name: `rag-chatbot`
5. Paste or upload the contents of `docker-compose.microservices.yml`
6. Click **Create**

### Method B: Using SSH Command Line (Recommended)

```bash
# SSH into QNAP
ssh admin@<QNAP_IP>
cd /share/Container/chatbot

# Start the stack (without Ollama - using external LM Studio)
docker-compose -f docker-compose.microservices.yml up -d

# OR with Ollama for local inference (requires more RAM)
docker-compose -f docker-compose.microservices.yml --profile ollama up -d
```

---

## Step 5: Adjust for QNAP Network

Modify `docker-compose.microservices.yml` for QNAP-specific networking:

```yaml
# Replace host.docker.internal references with your PC's actual IP
# In the chat-api service environment:
- LLM__LMSTUDIO__BASE_URL=http://192.168.1.xxx:1234/v1
```

Or create an override file:

```bash
cat > docker-compose.override.yml << 'EOF'
version: "3.9"
services:
  chat-api:
    environment:
      - LLM__LMSTUDIO__BASE_URL=http://192.168.1.xxx:1234/v1
  embedding-service:
    environment:
      - LMSTUDIO_URL=http://192.168.1.xxx:1234/v1
EOF
```

---

## Step 6: Pull Ollama Models (If Using Ollama Profile)

```bash
# After containers are running
docker exec rag-ollama ollama pull nomic-embed-text
docker exec rag-ollama ollama pull llama3.2:1b

# Optional: larger model for better quality
docker exec rag-ollama ollama pull llama3.2:3b
```

---

## Step 7: Verify Deployment

```bash
# Check container status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test health endpoints
curl http://localhost:8000/health
curl http://localhost:8020/api/v2/heartbeat  # ChromaDB

# Get auth token and test
TOKEN=$(curl -s http://localhost:8000/auth/login -X POST \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')

curl -s http://localhost:8000/chat -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello!","stream":false}'
```

---

## Step 8: Configure QNAP Firewall/Port Forwarding

In **QNAP Control Panel** → **Network & Virtual Switch** → **NAT**:

| Service | Internal Port | External Port |
|---------|---------------|---------------|
| Frontend | 80 | 80 |
| API | 8000 | 8000 |
| Grafana | 3001 | 3001 |

---

## Access Points After Deployment

| Service | URL |
|---------|-----|
| **Frontend** | `http://<QNAP_IP>/` |
| **API Docs** | `http://<QNAP_IP>:8000/docs` |
| **Grafana** | `http://<QNAP_IP>:3001` (admin/admin) |
| **Prometheus** | `http://<QNAP_IP>:9090` |

**Default Login:** `admin` / `admin123`

---

## QNAP-Specific Tips

1. **Persistent Storage:** Volumes are automatically persisted in `/share/Container/container-station-data/lib/docker/volumes/`

2. **Resource Limits:** If your NAS has limited RAM, use the simpler `docker-compose.yml` instead:

   ```bash
   docker-compose up -d  # Uses ~2GB RAM vs ~6GB for microservices
   ```

3. **Autostart on Boot:** In Container Station, enable "Auto Start" for the application

4. **Logs:** View logs via Container Station UI or:

   ```bash
   docker-compose -f docker-compose.microservices.yml logs -f chat-api
   ```

5. **Updates:**

   ```bash
   cd /share/Container/chatbot
   git pull
   docker-compose -f docker-compose.microservices.yml up -d --build
   ```

---

## Lightweight Option for Limited Hardware

If your QNAP has limited resources (4-8GB RAM), use the simpler standalone mode:

```bash
cd /share/Container/chatbot
docker-compose up -d  # Uses docker-compose.yml (not microservices)
```

This runs only 3 containers (MySQL, Backend, Frontend) and requires an external LLM provider like LM Studio running on another machine.

---

## Troubleshooting

### Container Station Not Starting Containers

```bash
# Check Docker daemon status
docker info

# View container logs
docker logs rag-chat-api
```

### Network Connectivity Issues

- Ensure QNAP firewall allows container traffic
- Verify LM Studio PC is reachable from QNAP: `ping <YOUR_PC_IP>`
- Check if ports are not blocked by your router

### MySQL Connection Errors

```bash
# Check MySQL container status
docker logs rag-mysql

# Verify MySQL is healthy
docker exec rag-mysql mysqladmin ping -h localhost -uroot -p
```

### Out of Memory

- Reduce number of services by using `docker-compose.yml` instead
- Disable monitoring stack (remove prometheus, grafana, loki, promtail services)
- Use smaller Ollama models (`llama3.2:1b` instead of `3b`)
