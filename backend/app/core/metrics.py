"""Prometheus metrics setup using prometheus-fastapi-instrumentator.

Provides a helper to attach default metrics and custom RAG/LLM metrics.
"""

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator.metrics import (
    request_size,
    response_size,
    latency,
    requests,
)

from app.config import settings


instrumentator: Instrumentator = Instrumentator()


def setup_metrics(app):
    """Configure Prometheus metrics for the FastAPI app."""
    if not settings.monitoring.metrics_enabled:
        return

    # Attach default metrics plus request/response sizes
    instrumentator \
        .add(requests()) \
        .add(latency()) \
        .add(request_size()) \
        .add(response_size()) \
        .instrument(app)

    # Expose /metrics endpoint
    instrumentator.expose(app)
