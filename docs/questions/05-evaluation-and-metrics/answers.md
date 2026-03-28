# 5 — Evaluation & Metrics — Answers

> All answers below are grounded in the **rag-ai** codebase.

---

## Q28. What metrics do you use to evaluate an agentic RAG system beyond simple answer accuracy?

Our system tracks multiple dimensions:

**1. Cost Tracking (`core/cost_tracker.py`):**
- Tracks `prompt_tokens`, `completion_tokens`, `total_tokens` per LLM call.
- Compares against OpenAI pricing to quantify savings from local models:
  ```python
  OPENAI_PRICING = {
      "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # per 1M tokens
      "gpt-4o": {"input": 2.50, "output": 10.00},
  }
  ```
- `cost_comparison_model = "gpt-4o-mini"` in config for baseline comparison.

**2. Retrieval Quality (`test_rag_quality.py` — 18.8 KB test suite):**
- Tests retrieval precision and relevance across various query types.
- Tests client isolation (no cross-client data leakage).
- Tests hybrid search vs. vector-only search quality.

**3. Latency:**
- `HybridSearch.search_with_stats()` tracks timing: `duration = time.time() - start_time`
- Returns `HybridSearchResult` with `bm25_candidates`, `vector_candidates`, `fused_candidates` counts.

**4. LangSmith Tracing:**
- Every agent step is traced with `@traceable` decorators.
- `langsmith_project = "agentic-rag"` enables per-project dashboards.
- Tracks latency, token usage, and errors per agent node.

**5. Faithfulness (implicit):**
- The synthesis prompt enforces source citation: `[Source: filename]`.
- The `sources` field in `AgentState` tracks which chunks were used.

---

## Q29. How do you measure and reduce hallucination rates?

**Measurement in our system:**

1. **Source attribution enforcement:** The synthesis prompt requires `[Source: filename]` citations. Responses without citations for factual claims indicate potential hallucination.
2. **RAG quality tests** (`test_rag_quality.py`): Automated tests verify that answers align with known document content.
3. **LangSmith traces:** Manual review of agent traces shows whether the synthesis agent’s response matches the retrieved chunks.

**Reduction strategies implemented:**

1. **Low synthesis temperature:** `temperature=0.35` reduces creative/hallucinatory outputs.
2. **Explicit grounding instructions:** The system prompt says: *"Answer based on the provided context when available"* and *"If the context doesn’t contain enough information, say so clearly."*
3. **Context windowing:** Only 6 chunks, each truncated to 600 chars, are passed to synthesis. This keeps the context focused and reduces the LLM’s temptation to interpolate.
4. **Cross-encoder reranking:** By surfacing the most relevant chunks (not just the most similar), the synthesis agent gets higher-quality context.
5. **Client context separation:** The system prompt specifies which client’s documents are being used, preventing cross-client confusion.

---

## Q30. Explain the RAGAS evaluation framework. Which metrics are most useful for agentic RAG?

RAGAS (Retrieval Augmented Generation Assessment) provides automated metrics:

| Metric | What it measures | Relevance to our system |
|---|---|---|
| **Faithfulness** | Is the answer supported by retrieved context? | Critical — our synthesis prompt enforces this via citation requirements |
| **Answer Relevancy** | Does the answer address the question? | Important — our intent classification + query rewriting improve this |
| **Context Precision** | Are retrieved chunks actually relevant? | Key — our reranker directly optimizes this |
| **Context Recall** | Are all relevant chunks retrieved? | Important — our hybrid search (BM25+vector) improves recall over vector-only |

**Most useful for our agentic system:**
1. **Faithfulness** — because our multi-agent pipeline adds complexity where hallucination can creep in.
2. **Context Precision** — because our reranker’s effectiveness can be directly measured.
3. **Answer Relevancy** — because our query rewriting step should improve this over raw queries.

Our `test_rag_quality.py` (18.8 KB) implements custom versions of these metrics tailored to our hybrid search pipeline.

---

## Q31. How do you build a regression test suite for an agentic RAG system?

Our test suite is structured in `tests/`:

```
tests/
├── test_rag_quality.py          # 18.8 KB - RAG retrieval quality tests
├── test_api_endpoints.py        # API endpoint tests
├── test_config.py               # Configuration tests
├── test_prompt_builder.py       # Prompt construction tests
├── integration/
│   ├── test_client_isolation.py  # 11.9 KB - Cross-client isolation
│   └── test_smoke.py            # 13.6 KB - E2E smoke tests
└── live/
    ├── conftest.py              # 8.1 KB - Live test fixtures
    └── test_endpoints.py        # 49.8 KB - Comprehensive endpoint tests
```

**Key regression strategies:**
1. **Golden test cases:** `test_rag_quality.py` contains known question-answer pairs against indexed documents.
2. **Client isolation tests:** `test_client_isolation.py` verifies that client A’s documents never appear in client B’s results.
3. **Smoke tests:** `test_smoke.py` exercises the full pipeline end-to-end.
4. **Configuration tests:** `test_config.py` ensures setting changes don’t break initialization.
5. **Live endpoint tests:** `test_endpoints.py` (49.8 KB!) tests every API endpoint with various inputs.

**Preventing silent degradation:**
- Embedding fingerprint verification catches model changes.
- CI runs `make test` on every commit.
- LangSmith dashboards show metric trends over time.

---

## Q32. What is "context precision" vs. "context recall"?

**Context Precision:** Of the chunks retrieved, what fraction is actually relevant to the query? Higher precision = less noise in context.

**Context Recall:** Of all relevant chunks in the knowledge base, what fraction was retrieved? Higher recall = fewer missed facts.

**How we instrument both:**

1. **Precision tracking:** The `HybridSearchResult` dataclass tracks:
   ```python
   bm25_candidates: int      # How many BM25 returned
   vector_candidates: int    # How many vector returned
   fused_candidates: int     # How many after RRF fusion
   ```
   Combined with `rerank_top_k = 5`, we can measure: of the 5 final chunks, how many were relevant?

2. **Recall tracking:** Our dual-source retrieval (client + global, `initial_fetch_k = 30` candidates each) casts a wide net. BM25 catches keyword matches that vector search misses, and vice versa — directly improving recall.

3. **Reranker metadata:** After reranking, each hit stores both `original_score` and `rerank_score`:
   ```python
   metadata = {
       "original_score": hit.score,
       "rerank_score": float(rerank_score),
   }
   ```
   This lets us analyze whether the reranker is improving precision (promoting relevant docs) or hurting recall (demoting relevant docs).

---

## Q33. How would you A/B test two different agent strategies in production?

Our architecture supports A/B testing through:

**1. Configuration-driven strategy selection:**
- `bm25_enabled: bool` toggles hybrid vs. vector-only search.
- `reranker_enabled: bool` toggles cross-encoder reranking.
- `bm25_weight` / `vector_weight` can be tuned per strategy.

**2. Multi-provider LLM support:** We can run Strategy A with `provider: "lmstudio"` and Strategy B with `provider: "groq"` simultaneously.

**3. LangSmith project-based comparison:**
- Route 50% of traffic to `langsmith_project = "agentic-rag-v1"` and 50% to `"agentic-rag-v2"`.
- Compare metrics (latency, faithfulness, user satisfaction) across projects.

**4. Cost tracking comparison:** The `CostTracker` can compare strategies by measuring tokens consumed per query under each approach.

**Implementation approach:**
- Add a `strategy` field to `ChatRequest`.
- In `ChatService.handle_chat()`, instantiate the appropriate orchestrator variant.
- Log strategy assignment + outcome metrics to LangSmith.
- Use statistical significance tests (not just averages) before declaring a winner.
