# 7 — Scalability & Production — Answers

> All answers below are grounded in the **rag-ai** codebase.

---

## Q39. How do you manage latency in an agentic RAG system with multiple retrieval rounds?

Our system manages latency through:

**1. Single-pass architecture:** The current LangGraph is a DAG with no iterative retrieval loops. A typical query traverses: `QueryAgent` → `RetrievalAgent` → `SynthesisAgent` (3 nodes, 2 LLM calls + 1 retrieval).

**2. Lightweight classification:** `QueryAgent` uses `max_tokens=256` and `temperature=0.1` for fast intent classification. Chitchat queries skip retrieval entirely.

**3. Concurrent retrieval:** The `RetrievalAgent` could search client and global collections in parallel (both use `asyncio` with `run_in_executor`).

**4. Semaphore-bounded reranking:** The reranker uses a `asyncio.Semaphore` to limit concurrent CPU-intensive cross-encoder runs:
```python
_max_concurrent = getattr(settings.rag, 'reranker_max_concurrent', 2)
RERANKER_SEMAPHORE = asyncio.Semaphore(_max_concurrent)
```
This prevents CPU saturation without blocking the event loop.

**5. Retrieval candidate limits:**
```python
initial_fetch_k: int = 30   # Candidates for reranking
rerank_top_k: int = 5       # Final results
```
Reranking 30 candidates (not 100+) keeps cross-encoder inference fast.

**6. Context truncation:** Synthesis receives max 6 chunks × 600 chars = ~3600 chars of context, keeping LLM input tokens low.

---

## Q40. What caching strategies do you use to reduce redundant retrievals and LLM calls?

Our system implements caching at multiple levels:

**1. Redis caching:**
```python
class RedisSettings(BaseModel):
    cache_ttl: int = 3600  # 1 hour cache TTL
```

**2. Singleton pattern for expensive objects:**
```python
# Reranker model loaded once
@lru_cache(maxsize=1)
def get_reranker() -> Reranker:

# Embedding function loaded once
@lru_cache(maxsize=1)
def get_embedding_function() -> EmbeddingFunction:

# HybridSearch instances cached per client
_hybrid_search_instances: Dict[str, HybridSearch] = {}
```

**3. Conversation summary caching:** Instead of re-processing the entire conversation history, the summarizer stores compressed summaries in Redis:
```python
summary_key_prefix: str = "summary:"
session_ttl: int = 86400  # 24 hours
```

**4. BM25 index persistence:** BM25 indices are persisted to disk (`./data/bm25`) and loaded on startup, avoiding expensive re-indexing.

**5. CrossEncoder lazy loading:** The reranker model is only loaded on first use:
```python
_cross_encoder = None
def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(model_name)
```

**What’s not yet implemented:** Semantic caching (embedding queries and checking similarity to cached queries before running retrieval).

---

## Q41. How do you implement observability and tracing for a multi-step agentic RAG pipeline?

Our system uses **LangSmith** for full observability:

**1. Per-node tracing with `@traceable` decorators:**
```python
@traceable(name="agent.query")
async def _query_node(self, state)

@traceable(name="agent.retrieval")
async def _retrieval_node(self, state)

@traceable(name="agent.tool")
async def _tool_node(self, state)

@traceable(name="agent.synthesis")
async def _synthesis_node(self, state)

@traceable(name="orchestrator.run")
async def run(self, ...)
```

Sub-operations are also traced:
```python
@traceable(name="query_agent.classify_intent")
@traceable(name="query_agent.rewrite_query")
@traceable(name="retrieval_agent.retrieve_with_global")
@traceable(name="synthesis_agent.generate_response")
```

**2. LangSmith configuration:**
```python
langsmith_tracing: bool = True
langsmith_project: str = "agentic-rag"
```
Enabled via env: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=...`

**3. Structured JSON logging (`core/logging.py`):**
- JSON-formatted logs for Loki/ELK ingestion.
- **Correlation ID tracking** via `ContextVar` across requests.
- Configurable file rotation (10MB per file, 30 backup files).
- Request metadata (user_id, action) in log context.

**4. Cost tracking as observability:**
The `CostTracker` (an `AsyncCallbackHandler`) is attached to every LLM call:
```python
self.llm = ChatOpenAI(..., callbacks=[self.cost_tracker])
```
It logs token usage and estimated costs per request.

**5. Health endpoint:**
`/health` route (`routes/health.py`) provides system status for monitoring.

---

## Q42. How would you auto-scale an agentic RAG service to handle bursty traffic?

Our Docker Compose architecture (`docker-compose.yml`) is designed for horizontal scaling:

**Current setup:**
```yaml
# docker-compose.yml services:
- Redis (sessions/memory)
- ChromaDB (vector store)
- Chat-API (FastAPI + LangGraph)
- Nginx (frontend + reverse proxy)
```

**Scaling strategy:**
1. **Stateless API layer:** The Chat-API is stateless — all state is in Redis and ChromaDB. Multiple API replicas can run behind Nginx.
2. **Redis as shared state:** Session buffer, summaries, and metadata are in Redis, accessible from any API replica.
3. **ChromaDB external mode:** `url: Optional[str]` in config supports remote ChromaDB, allowing the vector store to scale independently.
4. **Reranker semaphore per-instance:** Each API replica has its own `RERANKER_SEMAPHORE`, preventing local CPU overload.
5. **Production deployment:** `deploy/docker-compose.prod.yml` and `deploy/deploy.sh` support production infrastructure.
6. **Infrastructure as Code:** `infra/main.tf` provides Terraform-based infrastructure provisioning.

**Budget control:**
- `cost_tracking_enabled: bool = True` monitors per-request costs.
- `max_tokens = 4096` caps generation length.
- `max_steps = 10` bounds agent iterations.

---

## Q43. How do you handle rate limits and token budget management?

**Token budget management in our system:**

1. **Per-request token caps:**
   - `QueryAgent`: `max_tokens=256` (classification only)
   - `SynthesisAgent`: `max_tokens=4096` (full response)
   - Context window: `context_window=32000` tokens

2. **Memory-based token management:**
   ```python
   max_context_tokens: int = 4000   # Trigger summarization
   summary_target_tokens: int = 1000 # Target after summarization
   ```
   Auto-summarization compresses conversation history when it exceeds 4000 tokens.

3. **Cost tracking per call:**
   The `CostTracker` (`AsyncCallbackHandler`) records `prompt_tokens` and `completion_tokens` for every LLM invocation.

4. **LLM timeout:** `timeout: float = 120.0` prevents hung requests from consuming resources.

5. **Context truncation everywhere:**
   - Query rewrite context: `context[:500]`
   - Conversation summary in classification: `conversation_summary[:300]`
   - Retrieved chunks: `content[:600]` each, max 6 chunks

**Rate limiting:** Not explicitly implemented at the application level, but the reranker semaphore (`max_concurrent=2`) effectively rate-limits the most CPU-intensive operation.

---

## Q44. What is your strategy for graceful degradation?

Our system has **production-grade degradation** at multiple levels:

**1. Redis circuit breaker (`session_buffer.py`):**
```python
class RedisCircuitBreaker:
    # States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
```
When Redis is down:
- Circuit opens after 5 failures.
- Falls back to **memory-bounded in-memory storage** with LRU eviction.
- Automatically tests recovery every 30 seconds.
- After 3 successful calls, circuit closes.

**2. Embedding fallback (`embeddings.py`):**
```python
class NoOpEmbeddingFunction:
    """Returns zero vectors to avoid hard failures."""
```
If the embedding service is unavailable, the system returns zero vectors rather than crashing.

**3. Reranker fallback:**
```python
if not self.enabled or self.model is None:
    return hits[:top_k]  # Return original order
```
If the cross-encoder fails to load, retrieval continues with bi-encoder-only ranking.

**4. Orchestrator error boundary:**
```python
except Exception as e:
    return "I encountered an error processing your request.", []
```

**5. Per-agent error handling:** Each agent catches exceptions and returns graceful fallbacks rather than propagating failures.

**6. BM25 fallback:** If BM25 index is unavailable, `RetrievalAgent` falls back to vector-only search.

---

## Q45. How do you deploy embedding model updates without downtime?

**Our embedding fingerprinting system** (in `embeddings.py`) handles this:

1. **Fingerprint on write:** Every chunk stored in ChromaDB includes an `embedding_fingerprint` metadata field:
   ```
   "lmstudio:nomic-embed-text:768:true:1.0"
   ```

2. **Verify on read:** `verify_embedding_fingerprint()` compares stored vs. current fingerprint on startup.

3. **Mismatch detection:** `EmbeddingMismatchError` is raised if the model changed, preventing silent degradation from mixed embeddings.

4. **Per-client re-indexing:** Since collections are per-client (`ChromaClientVectorStore`), you can re-index one client at a time:
   - Create new collection with new embeddings.
   - Verify quality.
   - Swap the collection reference.
   - Delete old collection.

5. **BM25 is model-independent:** BM25 indices use raw text tokens, so they survive embedding model changes without any re-indexing.

6. **`verify_fingerprint=False` escape hatch:** The `ChromaClientVectorStore` constructor accepts this flag for migration scenarios where you intentionally mix old and new embeddings temporarily.
