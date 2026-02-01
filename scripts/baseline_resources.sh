#!/usr/bin/env bash
# Baseline script for measuring container CPU, memory, and volume usage.
# Run from repo root: ./scripts/baseline_resources.sh

set -e

echo "=== Docker system disk usage ==="
docker system df -v 2>/dev/null || docker system df

echo ""
echo "=== Named volumes (rag-*) ==="
for vol in rag-redis-data rag-chroma-data rag-bm25-data 2>/dev/null; do
  if docker volume inspect "$vol" &>/dev/null; then
    echo "--- $vol ---"
    docker volume inspect "$vol" --format '{{.Mountpoint}}'
    du -sh "$(docker volume inspect "$vol" --format '{{.Mountpoint}}' 2>/dev/null)" 2>/dev/null || echo "(run as root or with sudo to see size)"
  fi
done

echo ""
echo "=== Container stats (live snapshot) ==="
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null || true

echo ""
echo "=== Volume sizes via docker system df (volumes section) ==="
docker system df -v 2>/dev/null | grep -A 200 "Local Volumes" | head -30 || true
