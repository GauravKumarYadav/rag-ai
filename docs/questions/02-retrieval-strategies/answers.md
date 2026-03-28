# 2 — Retrieval Strategies — Answers

> All answers below are grounded in the **rag-ai** codebase.

---

## Q8. When an agent’s first retrieval attempt returns low-relevance results, what self-correction strategies can it employ?

In our current system, self-correction is **partially implemented**:

1. **Query Rewriting (pre-retrieval):** The `QueryAgent._rewrite_query()` rewrites conversational queries into standalone, retrieval-optimized queries before the first attempt. This preemptively improves relevance.
2. **Cross-Encoder Reranking (post-retrieval):** The `Reranker` (in `reranker.py`) uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rescore results. Even if initial vector/BM25 retrieval returns noisy candidates, the cross-encoder can surface the truly relevant ones.
3. **Dual-collection fallback:** The `RetrievalAgent._retrieve_with_global()` searches both the **client-specific** collection and the **global** collection, then merges results. If the client collection returns nothing, global results act as a fallback.
4. **Graceful admission of ignorance:** The `SynthesisAgent`’s system prompt instructs: *"If the context doesn’t contain enough information, say so clearly."* This prevents hallucination when retrieval fails.

**What’s not yet implemented but architecturally possible:**
- Iterative retrieval (re-query with decomposed sub-queries) — the LangGraph could add a feedback edge from synthesis back to retrieval.
- HyDE (generate a hypothetical answer, embed it, re-retrieve).

---

## Q9. Compare dense vector, sparse keyword, and hybrid retrieval. When would an agent dynamically switch?

Our system implements all three and makes this decision via configuration:

| Method | Implementation | Strengths |
|---|---|---|
| **Dense vector** | ChromaDB with `nomic-embed-text-v1.5` (768-dim) | Semantic similarity, paraphrase handling |
| **Sparse keyword (BM25)** | `rank_bm25.BM25Okapi` in `bm25_index.py` | Exact keyword matches, rare terms, factual lookups |
| **Hybrid (RRF)** | `hybrid_search.py` — Reciprocal Rank Fusion | Best of both; no score calibration needed |

From `config.py`:
```python
bm25_enabled: bool = True
bm25_weight: float = 0.4      # Weight for BM25 in RRF fusion
vector_weight: float = 0.6    # Weight for vector search in RRF fusion
```

The `RetrievalAgent._search_collection()` dynamically selects:
```python
if self.use_bm25:
    hybrid = get_hybrid_search(client_id=client_id)
    # Uses BM25 + Vector + RRF
else:
    retriever = get_retriever()
    # Uses vector-only search
```

**Dynamic switching scenarios:**
- For queries with specific proper nouns or codes (e.g., "policy ABC-123"), BM25 excels.
- For semantic/conceptual queries ("what is the refund process"), vector search dominates.
- Our hybrid approach with `bm25_weight=0.4` and `vector_weight=0.6` gives vector a slight edge while preserving BM25’s keyword strength.

---

## Q10. How do you implement adaptive chunking strategies, and how does chunk size impact agent reasoning?

Our chunking is implemented in `backend/app/processors/chunking.py` with rich configurability:

```python
# From config.py RAGSettings
chunk_size: int = 1200         # Target characters per chunk
chunk_overlap: int = 200       # Overlap between chunks
chunk_token_size: int = 512    # Target tokens (token mode)
chunk_token_overlap: int = 50  # Token overlap
min_chunk_tokens: int = 50     # Minimum chunk size
chunk_method: str = "char"     # "char" or "token" based
respect_headings: bool = True  # Break at markdown headers
respect_sentences: bool = True # Break at sentence boundaries
```

**Adaptive strategies in our codebase:**
1. **Heading-aware chunking** (`respect_headings=True`): Chunks align with markdown section headers from Docling output, keeping topically coherent chunks.
2. **Sentence-boundary respect** (`respect_sentences=True`): Avoids mid-sentence splits that confuse the reranker and synthesis agent.
3. **Content-hash IDs** (`use_content_hash_ids=True`): Deterministic chunk IDs via content hashing enable deduplication during re-ingestion.
4. **Two modes:** Character-based (default) and token-based chunking are both supported.

**Impact on agent reasoning:**
- Too small chunks (< 50 tokens, our `min_chunk_tokens`) lose context, forcing the synthesis agent to reason across many disconnected snippets.
- Too large chunks dilute relevance signals — the reranker has more noise to filter.
- Our default of 1200 chars (~300 tokens) with 200 char overlap strikes a balance: enough context for the synthesis agent while keeping retrieval precision high.

---

## Q11. Explain query decomposition in agentic RAG.

Query decomposition is the process of breaking a complex multi-part question into simpler sub-queries, retrieving for each, and merging results.

**In our system:** The `QueryAgent._rewrite_query()` performs a simpler form of this — it rewrites the query for retrieval but doesn’t explicitly decompose into sub-queries. The rewrite prompt:

```python
QUERY_REWRITE_PROMPT = """Rewrite the user's query to be more effective for document retrieval.
Guidelines:
- Remove conversational prefixes
- Make the query standalone (resolve references using conversation context)
- Focus on key terms and concepts
- Keep it concise but complete"""
```

**How full decomposition would work in our architecture:**
1. The `QueryAgent` would generate multiple sub-queries from a complex question.
2. Each sub-query would be passed to `RetrievalAgent._search_collection()` independently.
3. Results would be merged using the existing `_merge_results()` deduplication logic.
4. The `SynthesisAgent` would receive all chunks and synthesize across them.

The `AgentState.rewritten_query` field (currently a single string) would need to become a `List[str]` to support this.

---

## Q12. What is Hypothetical Document Embedding (HyDE), and when would an agent use it?

HyDE asks the LLM to generate a *hypothetical answer* to the query, embeds that hypothetical document, and uses it as the search vector instead of embedding the raw query.

**Why it helps:** The hypothetical document is in the same "language" as the indexed documents (declarative, detailed), so its embedding is often closer to relevant chunks than a short question embedding.

**When to use vs. direct lookup:**
- **HyDE:** When queries are abstract or conceptual ("What are best practices for X?") and the embedding model struggles with question-document asymmetry.
- **Direct embedding:** When queries are specific and contain the exact terms in the documents ("What is the refund policy for product Y?").

**In our system:** HyDE is not currently implemented. Our approach instead uses:
- **Query rewriting** to bridge the question-document gap.
- **Hybrid search (BM25 + vector)** to catch both semantic and lexical matches.
- **Cross-encoder reranking** to rescore based on joint query-document understanding.

These three together achieve much of what HyDE offers without the extra LLM call.

---

## Q13. How would you handle multi-hop retrieval?

Multi-hop retrieval requires chaining facts across multiple documents (e.g., "Who is the manager of the person who signed contract X?").

**Our system’s current approach:**
1. The `RetrievalAgent` searches both client and global collections and merges results — this naturally pulls from multiple documents.
2. The `SynthesisAgent` receives up to 6 chunks (`retrieved[:6]` in `_build_context`) and can reason across them.
3. The **knowledge graph module** (`backend/app/knowledge/`) is prepared for multi-hop: `graph_query.py` can traverse entity relationships, and `entity_extractor.py` detects named entities for graph construction.

**For true multi-hop, the architecture would need:**
- An iterative retrieval loop: retrieve → extract intermediate answer → re-retrieve with new context.
- The knowledge graph (`graph_store.py`) with entity types (`PERSON`, `ORG`, `DOCUMENT`) and relationship types (`AUTHORED`, `REFERENCES`, `WORKS_FOR`) is specifically designed for traversal-based multi-hop.
- Enabling `RAG__KNOWLEDGE_GRAPH_ENABLED=true` activates this subsystem.

---

## Q14. Describe a re-ranking strategy an agent can use after initial retrieval.

Our system implements **cross-encoder reranking** in `backend/app/rag/reranker.py`:

```python
class Reranker:
    def rerank(self, query, hits, top_k=None):
        # Create query-document pairs
        pairs = [(query, hit.content) for hit in hits]
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        # Sort by rerank score (higher is better)
        scored_hits.sort(key=lambda x: x.score, reverse=True)
```

**How cross-encoders fit in:**
- **Bi-encoders** (our embedding model) encode query and document separately — fast but less accurate.
- **Cross-encoders** (`ms-marco-MiniLM-L-6-v2`, ~22M params) encode query+document *jointly* — slower but significantly more accurate.
- We use bi-encoders for initial retrieval (fast, `fetch_k=30` candidates) and cross-encoders for reranking (accurate, `top_k=5` final results).

Additionally, our system includes an **MMR (Maximal Marginal Relevance) selector** in the same file:
```python
class MMRSelector:
    # MMR = λ * relevance - (1-λ) * max_similarity_to_selected
    # Balances relevance with diversity
```
This prevents returning 5 near-duplicate chunks, using either Jaccard text similarity or embedding cosine similarity for diversity.

---

## Q15. What is the role of metadata filtering, and how can an agent learn to apply filters dynamically?

Metadata filtering restricts retrieval to documents matching specific criteria (client, date, source file, page number).

**In our system:**
- **Client isolation is the primary filter:** The `RetrievalAgent` searches per-client collections (`client_{id}_docs`) and global collections separately. This is enforced at the collection level in ChromaDB, not via `where` clauses.
- **Chunk metadata** (from `ChunkMetadata` in `chunking.py`) includes: `client_id`, `source_filename`, `page_number`, `section_heading`, `doc_id`, `chunk_index`.
- The `HybridSearch.search()` accepts an optional `where` parameter for Chroma metadata filtering.
- The `BM25Index.search()` also supports `where` filters.

**Dynamic filter learning:**
Currently, filters are statically applied (client_id-based). An agent could learn to apply filters dynamically by:
1. Having the `QueryAgent` extract filter hints from the query (e.g., "in the 2024 report" → `{"source_filename": {"$contains": "2024"}}`).
2. Passing these as the `where` parameter through `AgentState`.
3. The `RetrievalAgent` would apply them to both `vector_store.query()` and `bm25_index.search()`.
