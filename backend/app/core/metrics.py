"""Prometheus metrics setup using prometheus-fastapi-instrumentator.

Provides a helper to attach default metrics and custom RAG/LLM metrics.
"""

from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator.metrics import (
    request_size,
    response_size,
    latency,
    requests,
)

from app.config import settings


# =============================================================================
# RAG-SPECIFIC METRICS
# =============================================================================

# Retrieval metrics
rag_retrieval_duration_seconds = Histogram(
    "rag_retrieval_duration_seconds",
    "Duration of RAG retrieval operations",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
    labelnames=["client_id", "search_type"],
)

rag_retrieval_results_total = Histogram(
    "rag_retrieval_results_total",
    "Number of results returned by retrieval",
    buckets=(0, 1, 2, 3, 5, 10, 20, 50),
    labelnames=["client_id"],
)

# Client access control metrics
rag_client_access_denied_total = Counter(
    "rag_client_access_denied_total",
    "Number of times client access was denied due to authorization",
    labelnames=["client_id", "user_id"],
)

rag_cross_client_filter_applied_total = Counter(
    "rag_cross_client_filter_applied_total",
    "Number of times the hard client_id filter was applied to retrieval",
    labelnames=["client_id"],
)

# Citation and verification metrics
rag_citations_per_response = Histogram(
    "rag_citations_per_response",
    "Number of citations included in each response",
    buckets=(0, 1, 2, 3, 5, 10, 20),
)

rag_verification_failures_total = Counter(
    "rag_verification_failures_total",
    "Number of answer verification failures (potential hallucinations)",
    labelnames=["failure_type"],
)

rag_ungrounded_claims_total = Counter(
    "rag_ungrounded_claims_total",
    "Number of claims made without supporting evidence",
)

rag_citation_coverage_ratio = Histogram(
    "rag_citation_coverage_ratio",
    "Ratio of cited claims to total claims in responses",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Pipeline stage metrics
rag_stage_skipped_total = Counter(
    "rag_stage_skipped_total",
    "Number of times a pipeline stage was skipped",
    labelnames=["stage", "reason"],
)

rag_stage_executed_total = Counter(
    "rag_stage_executed_total",
    "Number of times a pipeline stage was executed",
    labelnames=["stage"],
)

rag_rerank_skipped_total = Counter(
    "rag_rerank_skipped_total",
    "Number of times reranking was skipped due to high confidence",
)

rag_rerank_duration_seconds = Histogram(
    "rag_rerank_duration_seconds",
    "Duration of reranking operations",
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0),
)

rag_verification_skipped_total = Counter(
    "rag_verification_skipped_total",
    "Number of times answer verification was skipped",
)

# Knowledge Graph metrics
rag_kg_expansion_triggered_total = Counter(
    "rag_kg_expansion_triggered_total",
    "Number of times KG expansion was triggered",
    labelnames=["reason"],
)

rag_kg_entities_found_total = Histogram(
    "rag_kg_entities_found_total",
    "Number of entities found during KG expansion",
    buckets=(0, 1, 2, 3, 5, 10, 20),
)

# Embedding metrics
rag_embedding_mismatch_total = Counter(
    "rag_embedding_mismatch_total",
    "Number of embedding model mismatch detections (requires reindex)",
)

rag_embedding_duration_seconds = Histogram(
    "rag_embedding_duration_seconds",
    "Duration of embedding generation",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    labelnames=["batch_size"],
)

# Chunking metrics
rag_chunks_created_total = Counter(
    "rag_chunks_created_total",
    "Total number of chunks created during document ingestion",
    labelnames=["client_id", "doc_type"],
)

rag_chunk_size_tokens = Histogram(
    "rag_chunk_size_tokens",
    "Size of chunks in tokens",
    buckets=(50, 100, 200, 300, 400, 500, 600, 800, 1000),
)

# Context compression metrics
rag_compression_ratio = Histogram(
    "rag_compression_ratio",
    "Ratio of compressed to original context size",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

rag_facts_extracted_total = Histogram(
    "rag_facts_extracted_total",
    "Number of facts extracted by context compressor",
    buckets=(0, 1, 2, 3, 5, 10, 15, 20),
)

# Intent classification metrics
rag_intent_classification_total = Counter(
    "rag_intent_classification_total",
    "Count of queries by classified intent",
    labelnames=["intent"],
)

# LLM metrics
llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "Duration of LLM API requests",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
    labelnames=["provider", "model"],
)

llm_tokens_used_total = Counter(
    "llm_tokens_used_total",
    "Total tokens used in LLM requests",
    labelnames=["provider", "token_type"],
)

llm_request_errors_total = Counter(
    "llm_request_errors_total",
    "Total LLM request errors",
    labelnames=["provider", "error_type"],
)

# Active sessions gauge
rag_active_sessions = Gauge(
    "rag_active_sessions",
    "Number of active chat sessions",
)


# =============================================================================
# INSTRUMENTATION SETUP
# =============================================================================

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


# =============================================================================
# HELPER FUNCTIONS FOR RECORDING METRICS
# =============================================================================

def record_retrieval_duration(client_id: str, search_type: str, duration: float):
    """Record the duration of a retrieval operation."""
    rag_retrieval_duration_seconds.labels(
        client_id=client_id or "global",
        search_type=search_type,
    ).observe(duration)


def record_retrieval_results(client_id: str, count: int):
    """Record the number of results from a retrieval."""
    rag_retrieval_results_total.labels(
        client_id=client_id or "global",
    ).observe(count)


def record_client_access_denied(client_id: str, user_id: str):
    """Record a client access denial."""
    rag_client_access_denied_total.labels(
        client_id=client_id,
        user_id=user_id,
    ).inc()


def record_cross_client_filter(client_id: str):
    """Record that the hard client filter was applied."""
    rag_cross_client_filter_applied_total.labels(
        client_id=client_id or "global",
    ).inc()


def record_citations(count: int):
    """Record the number of citations in a response."""
    rag_citations_per_response.observe(count)


def record_verification_failure(failure_type: str):
    """Record a verification failure."""
    rag_verification_failures_total.labels(failure_type=failure_type).inc()


def record_ungrounded_claim():
    """Record an ungrounded claim."""
    rag_ungrounded_claims_total.inc()


def record_citation_coverage(ratio: float):
    """Record the citation coverage ratio."""
    rag_citation_coverage_ratio.observe(ratio)


def record_stage_skipped(stage: str, reason: str):
    """Record that a pipeline stage was skipped."""
    rag_stage_skipped_total.labels(stage=stage, reason=reason).inc()


def record_stage_executed(stage: str):
    """Record that a pipeline stage was executed."""
    rag_stage_executed_total.labels(stage=stage).inc()


def record_rerank_skipped():
    """Record that reranking was skipped."""
    rag_rerank_skipped_total.inc()


def record_rerank_duration(duration: float):
    """Record the duration of a rerank operation."""
    rag_rerank_duration_seconds.observe(duration)


def record_kg_expansion(reason: str):
    """Record that KG expansion was triggered."""
    rag_kg_expansion_triggered_total.labels(reason=reason).inc()


def record_intent(intent: str):
    """Record the classified intent of a query."""
    rag_intent_classification_total.labels(intent=intent).inc()


def record_llm_request(provider: str, model: str, duration: float):
    """Record an LLM request."""
    llm_request_duration_seconds.labels(provider=provider, model=model).observe(duration)


def record_llm_tokens(provider: str, prompt_tokens: int, completion_tokens: int):
    """Record LLM token usage."""
    llm_tokens_used_total.labels(provider=provider, token_type="prompt").inc(prompt_tokens)
    llm_tokens_used_total.labels(provider=provider, token_type="completion").inc(completion_tokens)


def record_llm_error(provider: str, error_type: str):
    """Record an LLM error."""
    llm_request_errors_total.labels(provider=provider, error_type=error_type).inc()


def record_embedding_mismatch():
    """Record an embedding model mismatch."""
    rag_embedding_mismatch_total.inc()


def record_chunks_created(client_id: str, doc_type: str, count: int):
    """Record chunks created during ingestion."""
    rag_chunks_created_total.labels(
        client_id=client_id or "global",
        doc_type=doc_type,
    ).inc(count)


def set_active_sessions(count: int):
    """Set the number of active sessions."""
    rag_active_sessions.set(count)
