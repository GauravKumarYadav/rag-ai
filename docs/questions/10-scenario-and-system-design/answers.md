# 10 — Scenario & System Design — Answers

> All answers below are grounded in the **rag-ai** codebase.

---

## Q61. Design an agentic RAG system for a legal firm with 10 million case documents and strict citation requirements.

**Adapting our architecture for this scenario:**

**1. Vector store upgrade:**
- Replace ChromaDB with **Qdrant or Weaviate** for horizontal scalability at 10M+ documents.
- Our `VectorStoreBase` abstract interface (`rag/base.py`) makes this a drop-in swap.
- Use IVF-PQ indexing instead of HNSW to manage memory at scale.

**2. Multi-tenant isolation (already built):**
- Our per-client collection architecture (`client_{id}_docs`) maps to per-law-firm or per-case isolation.
- Separate ChromaDB collections per client prevent cross-case data leakage.

**3. Strict citation (already implemented):**
- Synthesis prompt: *"Cite sources using [Source: filename] format."*
- `ChunkMetadata` includes `page_number` and `section_heading` for pinpoint citations.
- The `sources` list in `AgentState` provides structured citation data.
- Enhancement: Add `paragraph_index` (already in `ChunkMetadata`) and exact quote extraction.

**4. Chunking for legal documents:**
- Increase `chunk_size` to ~2000 chars for legal language that requires broader context.
- Enable `respect_headings` to align chunks with legal document sections (clauses, articles).
- Use the knowledge graph (`entity_extractor.py`) to extract parties, dates, amounts, and case references.

**5. Knowledge graph for case relationships:**
- Enable `RAG__KNOWLEDGE_GRAPH_ENABLED=true`.
- Entity types: `PERSON` (parties, judges), `ORG` (firms, courts), `DATE` (filing dates), `DOCUMENT` (case references).
- Relationship types: `REFERENCES` (case citations), `SIGNED`, `AUTHORED`.
- `graph_query.py` enables multi-hop traversal: "Find all cases cited by cases involving Company X."

**6. Retrieval strategy:**
- Hybrid search (BM25 + vector) is critical for legal — legal queries often contain specific statute numbers (BM25 strength) and conceptual similarity (vector strength).
- Cross-encoder reranking ensures the most relevant passages surface.

**7. Observability:**
- LangSmith tracing for audit trails.
- Structured JSON logging with correlation IDs for compliance.

---

## Q62. A user reports the agent is confidently returning outdated information. Diagnosis and fix.

**Diagnosis using our system’s tools:**

**Step 1 — Trace the request in LangSmith:**
- Find the trace for the problematic query in the `agentic-rag` project.
- Check the `retrieval_agent.retrieve_with_global` span: which chunks were retrieved?
- Check `retrieved_chunks` metadata: `source_filename`, `uploaded_at`, `page_number`.

**Step 2 — Verify document currency:**
- Use the `/documents` API endpoint to list all documents with their upload dates.
- Check if outdated documents are still in the knowledge base.
- Verify the `embedding_fingerprint` matches current config (stale embeddings from an old model could surface wrong results).

**Step 3 — Check BM25 index freshness:**
- BM25 indices persist at `./data/bm25`. If a document was deleted from ChromaDB but the BM25 index wasn’t updated, stale results could surface through BM25.
- The `HybridSearch.delete()` method must be called for both indices.

**Fix:**

1. **Delete outdated documents** via `DELETE /documents/{doc_id}` API — this should remove from both ChromaDB and BM25.
2. **Upload updated documents** — content-hash IDs ensure only changed chunks are re-embedded.
3. **Add timestamp-based recency filtering:**
   ```python
   where = {"uploaded_at": {"$gte": "2025-01-01"}}
   ```
   Pass this to `HybridSearch.search()` to prefer recent documents.
4. **BM25 index rebuild:** If the index is corrupted, clear `./data/bm25` and re-ingest.

---

## Q63. P95 latency spiked from 3s to 12s after deployment. Debugging process.

**Step 1 — Identify the slow node:**
- Check LangSmith traces for recent requests. The `@traceable` decorators on each node (`agent.query`, `agent.retrieval`, `agent.synthesis`) show per-node latency.
- Hypothesis: which node’s duration increased?

**Step 2 — Common culprits in our system:**

| Suspect | Symptom | Check |
|---|---|---|
| **LLM provider slowdown** | `agent.query` or `agent.synthesis` slow | Check LM Studio/Groq response times |
| **Reranker overload** | `agent.retrieval` slow | Check `RERANKER_SEMAPHORE` contention (max_concurrent=2) |
| **ChromaDB index size** | `agent.retrieval` slow | Check collection size; HNSW degrades at scale |
| **BM25 index corruption** | `agent.retrieval` slow | Check BM25 index file size at `./data/bm25` |
| **Redis latency** | Overall slow | Check circuit breaker state; Redis memory usage |
| **Embedding model change** | `agent.retrieval` slow | `EmbeddingMismatchError` in logs? |
| **New documents ingested** | `agent.retrieval` slow | Large batch ingestion can bloat indices |

**Step 3 — System-level checks:**
- `docker stats` to check container CPU/memory.
- Check if `DOCLING_NUM_THREADS = 4` is competing with reranker for CPU.
- Check `HybridSearchResult` stats: did `bm25_candidates` or `vector_candidates` spike?

**Step 4 — Quick mitigations:**
- Reduce `initial_fetch_k` from 30 to 15 (fewer candidates to rerank).
- Increase `reranker_max_concurrent` if CPU allows.
- Disable BM25 temporarily (`bm25_enabled = false`) to isolate.
- Check LLM `timeout: float = 120.0` — maybe reduce to 30s to fail fast.

---

## Q64. Design an agentic RAG system supporting 50+ languages.

**Adapting our architecture:**

**1. Multilingual embeddings:**
- Replace `nomic-embed-text-v1.5` with a multilingual model like `multilingual-e5-large` or `cohere-multilingual-v3`.
- Our config makes this simple: `RAG__EMBEDDING_MODEL=multilingual-e5-large`.
- Embedding fingerprinting will detect the change and flag re-indexing.

**2. Language-aware chunking:**
- Our current sentence boundary detection (`respect_sentences=True`) works for Latin-script languages.
- For CJK languages (no spaces), switch to token-based chunking: `chunk_method = "token"`.
- For RTL languages (Arabic, Hebrew), ensure Docling handles bidirectional text.

**3. Multilingual BM25:**
- Our BM25 stopword list is English-only. Add per-language stopword lists.
- BM25 tokenization (whitespace-based) needs adaptation for CJK.
- Consider replacing `rank_bm25` with a language-aware tokenizer.

**4. Query language detection:**
- Add a language detection step in `QueryAgent` before query rewriting.
- Route to language-specific rewrite prompts.
- The multilingual LLM (Qwen3-VL supports 29+ languages) handles cross-lingual retrieval.

**5. Per-language quality monitoring:**
- Track retrieval quality metrics per language in LangSmith.
- Languages with low retrieval quality need:
  - More training data for embedding fine-tuning.
  - Language-specific BM25 configuration.
  - Dedicated evaluation test suites.

**6. Cross-lingual retrieval:**
- User asks in French, documents are in English: multilingual embeddings map both to a shared space.
- The synthesis agent responds in the user’s language (LLM handles this naturally).

---

## Q65. Integrating real-time web search as a fallback without sacrificing groundedness.

**Design within our architecture:**

**1. Add web search as a tool:**
- Add `"web_search"` to `QueryAgent._detect_tool()` and `ToolAgent`.
- Trigger when: retrieval returns zero results OR all rerank scores are below a threshold.

**2. Conditional routing:**
```python
def _route_after_retrieval(self, state):
    if not state.get("retrieved_chunks") or all_low_quality(state):
        return "web_search"  # fallback
    return "synthesis"
```

**3. Groundedness preservation:**
- **Source labeling:** Web results are tagged with `{"collection_type": "web", "source": url}` in metadata.
- **Synthesis prompt update:** Add instruction: *"For web sources, cite the URL and note that this information is from the web, not from the knowledge base."*
- **Confidence hierarchy:** Knowledge base results are preferred over web results. The `_merge_results()` prioritization (client > global) extends to: client > global > web.
- **Freshness indicator:** Web results include a retrieval timestamp.

**4. Safety:**
- Whitelist allowed search domains.
- Rate limit web search calls.
- Cache web search results in Redis (`cache_ttl = 3600`).
- Never use web results without attribution.

**5. User transparency:**
- The response clearly distinguishes: *"Based on your documents: ..."* vs. *"From web search (not in your knowledge base): ..."*
