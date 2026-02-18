#!/usr/bin/env bash
# =============================================================================
# Deploy script — runs on the Oracle Cloud VM during CI/CD or manually
# Usage: cd /opt/rag-ai && bash deploy/deploy.sh
# =============================================================================
set -euo pipefail

APP_DIR="/opt/rag-ai"
COMPOSE_FILES="-f docker-compose.yml -f deploy/docker-compose.prod.yml"
HEALTH_URL="http://localhost/health"
MAX_RETRIES=30
RETRY_INTERVAL=10

cd "$APP_DIR"

echo "=========================================="
echo " RAG AI - Deploying..."
echo "=========================================="

# --- Pull latest code ---
echo "[1/5] Pulling latest code from GitHub..."
git fetch origin main
git reset --hard origin/main

# --- Ensure .env exists ---
echo "[2/5] Checking .env..."
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Copy from .env.production.example and configure."
    echo "  cp .env.production.example .env && nano .env"
    exit 1
fi

# --- Build and deploy ---
echo "[3/5] Building and starting containers..."
docker compose $COMPOSE_FILES build --pull
docker compose $COMPOSE_FILES up -d --remove-orphans

# --- Cleanup old images ---
echo "[4/5] Pruning unused Docker images..."
docker image prune -f

# --- Health check ---
echo "[5/5] Waiting for services to become healthy..."
for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo "Health check passed! Deployment successful."
        echo ""
        echo "=========================================="
        echo " Deployment complete!"
        echo "=========================================="
        docker compose $COMPOSE_FILES ps
        exit 0
    fi
    echo "  Attempt $i/$MAX_RETRIES — waiting ${RETRY_INTERVAL}s..."
    sleep "$RETRY_INTERVAL"
done

echo "ERROR: Health check failed after $((MAX_RETRIES * RETRY_INTERVAL))s"
echo "Check logs with: docker compose $COMPOSE_FILES logs"
exit 1
