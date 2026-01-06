# Monitoring & Observability Guide

This guide covers the monitoring stack including Prometheus metrics, structured logging with Loki, and Grafana dashboards.

---

## Overview

The application includes a full observability stack:

| Component | Purpose | Default Port |
|-----------|---------|--------------|
| **Prometheus** | Metrics collection & alerting | 9090 |
| **Loki** | Log aggregation | 3100 |
| **Promtail** | Log shipping to Loki | - |
| **Grafana** | Dashboards & visualization | 3001 |

---

## Quick Start

### 1. Start the Monitoring Stack

```bash
cd docker/monitoring
podman-compose up -d   # or docker-compose up -d
```

### 2. Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3001 | admin / admin |
| Prometheus | http://localhost:9090 | - |
| Backend Metrics | http://localhost:8000/metrics | - |

### 3. Stop the Stack

```bash
cd docker/monitoring
podman-compose down
```

---

## Prometheus Metrics

The backend exposes metrics at `/metrics` using `prometheus-fastapi-instrumentator`.

### Default HTTP Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | Request latency |
| `http_requests_in_progress` | Gauge | Current active requests |

### Custom Application Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `rag_retrieval_duration_seconds` | Histogram | client_id | RAG retrieval latency |
| `rag_documents_retrieved` | Histogram | client_id | Documents per query |
| `llm_request_duration_seconds` | Histogram | model, provider | LLM API latency |
| `llm_tokens_used` | Counter | model, type | Token usage |

### Scrape Configuration

Prometheus is configured to scrape the backend at 15s intervals:

```yaml
# docker/monitoring/prometheus/prometheus.yml
scrape_configs:
  - job_name: 'rag-ai-backend'
    static_configs:
      - targets: ['host.docker.internal:8000']
    scrape_interval: 15s
```

---

## Structured Logging

Logs are output in JSON format for machine parsing and shipped to Loki.

### Log Format

```json
{
  "timestamp": "2026-01-03T12:00:00.000Z",
  "level": "INFO",
  "logger": "app.api.routes.chat",
  "message": "Chat request processed",
  "correlation_id": "abc123",
  "client_id": "client-456",
  "duration_ms": 234.5
}
```

### Log Levels

| Level | When to Use |
|-------|-------------|
| `DEBUG` | Detailed debugging info |
| `INFO` | Normal operations |
| `WARNING` | Something unexpected but handled |
| `ERROR` | Errors that need attention |
| `CRITICAL` | System failures |

### Configuration

```python
# backend/app/config.py
class LoggingSettings(BaseModel):
    level: str = "INFO"
    log_dir: str = "./logs"
    log_file: str = "app.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 30
    json_format: bool = True
```

### Correlation IDs

Every request gets a unique correlation ID for tracing:

```python
# Automatically added via CorrelationIdMiddleware
logger.info("Processing request", extra={"correlation_id": get_correlation_id()})
```

---

## Loki Log Aggregation

Promtail ships logs from `./logs/` to Loki.

### Querying Logs in Grafana

1. Go to Grafana → Explore
2. Select Loki data source
3. Use LogQL queries:

```logql
# All logs from the app
{job="rag-ai"}

# Errors only
{job="rag-ai"} |= "ERROR"

# Filter by correlation ID
{job="rag-ai"} | json | correlation_id="abc123"

# Chat requests over 500ms
{job="rag-ai"} | json | duration_ms > 500
```

### Retention

Logs are retained for 7 days by default:

```yaml
# docker/monitoring/loki/loki-config.yml
limits_config:
  retention_period: 168h  # 7 days
```

---

## Grafana Dashboards

### Pre-configured Dashboards

1. **System Overview** - Request rates, error rates, latency
2. **RAG Performance** - Retrieval times, document counts
3. **LLM Metrics** - Token usage, model latency

### Creating Custom Dashboards

1. Login to Grafana at http://localhost:3001
2. Go to Dashboards → New Dashboard
3. Add panels with Prometheus or Loki queries

### Example Prometheus Queries

```promql
# Request rate (per second)
rate(http_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# RAG retrieval latency
histogram_quantile(0.95, rate(rag_retrieval_duration_seconds_bucket[5m]))
```

---

## Alerting

### Prometheus Alerts

Create alert rules in `docker/monitoring/prometheus/alerts.yml`:

```yaml
groups:
  - name: rag-ai
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected

      - alert: SlowResponses
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: 95th percentile latency above 5s
```

### Grafana Alerts

1. Edit any panel
2. Go to Alert tab
3. Create alert conditions
4. Configure notification channels (email, Slack, etc.)

---

## Troubleshooting

### Metrics Not Showing

1. Check backend is running: `curl http://localhost:8000/metrics`
2. Check Prometheus targets: http://localhost:9090/targets
3. Verify firewall allows container→host communication

### Logs Not Appearing in Loki

1. Check Promtail is running: `podman logs promtail`
2. Verify log files exist: `ls -la ./logs/`
3. Check Loki health: `curl http://localhost:3100/ready`

### Grafana Can't Connect to Data Sources

1. Verify data source URLs use `host.docker.internal` or container names
2. Check network connectivity between containers
3. Test data source connection in Grafana UI

---

## Production Considerations

### Resource Limits

```yaml
# docker-compose.yml
services:
  prometheus:
    deploy:
      resources:
        limits:
          memory: 1G
  loki:
    deploy:
      resources:
        limits:
          memory: 512M
```

### Persistent Storage

```yaml
volumes:
  prometheus_data:
    driver: local
  loki_data:
    driver: local
  grafana_data:
    driver: local
```

### External Storage

For production, consider:
- **Prometheus**: Remote write to Thanos/Cortex
- **Loki**: S3/GCS backend for chunks
- **Grafana**: PostgreSQL/MySQL for dashboards
