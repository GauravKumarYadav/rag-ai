# 3 — Vector Stores & Indexing — Answers

> All answers below are grounded in the **rag-ai** codebase.

---

## Q16. How do you choose between vector databases for a production agentic RAG system?

Our system uses **ChromaDB** as the primary vector store (`backend/app/rag/chroma_store.py`). The codebase also includes a **Pinecone stub** (`pinecone_store.py`) for future cloud deployment.

**Criteria that drove our choice:**

| Criteria | ChromaDB (our choice) | Pinecone | Weaviate | Qdrant |
|---|---|---|---|---|
| **Local/embedded** | ✅ Runs in-process or via Docker | ❌ Cloud-only (managed) | ✅ Self-hosted option | ✅ Self-hosted option |
| **Privacy** | ✅ All data stays local | ❌ Data in vendor cloud | ✅ Self-hosted | ✅ Self-hosted |
| **Setup complexity** | Minimal — pip install + Docker | API key + cloud setup | Docker Compose | Docker Compose |
| **Multi-tenancy** | Collection-per-client (our approach) | Namespace-per-tenant | Native multi-tenancy | Collection-per-tenant |
| **Metadata filtering** | ✅ `where` clause support | ✅ Rich filtering | ✅ GraphQL filters | ✅ Payload filtering |
| **Scalability** | Single-node (good for <10M vectors) | Managed, scales to billions | Horizontally scalable | Horizontally scalable |

**Our architecture note** (from `chroma_store.py` header):
> *"ChromaDB is a lightweight, embedded vector database ideal for local development, single-machine deployments, privacy-focused applications. For production with high availability, consider Pinecone, Weaviate, or Qdrant."*

**What matters most:** For our use case (per-client isolated document collections, local-first deployment, privacy), ChromaDB’s simplicity and embedded mode win. For enterprise scale, we’d migrate to Qdrant or Weaviate via the `VectorStoreBase` abstract interface.

---

## Q17. Explain HNSW, IVF-PQ, and brute-force ANN indexes. When does the choice matter?

- **HNSW (Hierarchical Navigable Small World):** Graph-based index. High recall, fast queries, but high memory usage. ChromaDB uses HNSW by default.
- **IVF-PQ (Inverted File with Product Quantization):** Clusters vectors into partitions, then compresses with PQ. Lower memory, but trades recall for speed. Better for >10M vectors.
- **Brute-force:** Exact nearest neighbor — scans all vectors. Perfect recall but O(n) query time.

**In our system:** ChromaDB uses **HNSW** under the hood. With our embedding dimension of 768 (`embedding_dimension: int = 768` in `config.py`), this is appropriate for:
- Collections up to ~1M vectors per client.
- Our `initial_fetch_k = 30` candidate retrieval — HNSW handles this in milliseconds.

**When the choice materially impacts agent performance:**
- At < 100K vectors: Brute-force or HNSW both work fine. Our system falls here.
- At 1M-10M vectors: HNSW’s memory becomes expensive; IVF-PQ saves ~4x memory.
- At > 10M vectors: IVF-PQ or disk-based indexes (Qdrant’s HNSW+mmap) become necessary.
- The reranker compensates for ANN recall loss: even if HNSW or IVF-PQ misses a relevant doc in initial retrieval, it won’t matter if the doc isn’t in the top-30 candidates anyway.

---

## Q18. How do you handle incremental index updates without full re-indexing?

Our system handles incremental updates at multiple levels:

**1. ChromaDB (vector store):**
- Documents are added with `collection.add()` or `collection.upsert()` — ChromaDB handles incremental HNSW graph updates.
- Content-hash IDs (`use_content_hash_ids = True` in config) make upserts idempotent — re-uploading the same document produces the same chunk IDs, so unchanged chunks are naturally deduplicated.

**2. BM25 Index:**
- `BM25Index.add_documents()` appends new documents and rebuilds the BM25 scoring model.
- `BM25Index.delete()` removes documents by ID or metadata filter.
- The index persists to disk at `./data/bm25` and reloads on startup.

**3. Hybrid Search sync:**
- `HybridSearch.add_documents()` updates both BM25 and vector indices.
- `HybridSearch.delete()` removes from BM25; vector store deletion is handled separately.

**4. Hot reload:** From `README.md`: *"Documents immediately available after upload"* — no server restart needed.

**For deletions and modifications:**
- Document deletion removes chunks from both ChromaDB and BM25.
- Document modification = delete old chunks + add new chunks (via content-hash ID change detection).

---

## Q19. Strategies for multi-tenancy in a shared vector store with per-tenant retrieval isolation.

Our system implements **collection-per-client isolation** (from `ARCHITECTURE.md`):

```
Global Collection:  global_docs
Client Collection:  client_{id}_docs
```

**Implementation details:**

1. **Separate ChromaDB collections:** `ChromaClientVectorStore` (in `chroma_store.py`) creates a collection per client with sanitized names via `sanitize_collection_name()`:
   ```python
   def sanitize_collection_name(name: str) -> str:
       # Rules: 3-63 chars, alphanumeric + underscores + hyphens
       # Includes hash suffix to prevent collisions
       # "client-1" -> "client_1_a1b2c3d4"
   ```

2. **Separate BM25 indices:** `get_bm25_index(client_id=client_id)` creates per-client BM25 indexes.

3. **Separate HybridSearch instances:** `get_hybrid_search(client_id=client_id)` caches per-client instances:
   ```python
   _hybrid_search_instances: Dict[str, HybridSearch] = {}
   ```

4. **Retrieval-time isolation:** `RetrievalAgent._retrieve_with_global()` searches client and global collections independently, then merges with client results prioritized.

5. **API-level enforcement:** JWT auth + client_id validation on all document operations.

This approach is **stronger than metadata-filter-based isolation** because a bug in filter logic can’t leak data across tenants — collections are physically separate.

---

## Q20. How do you version and roll back an embedding index when your embedding model changes?

Our system implements **embedding fingerprinting** (in `embeddings.py`):

```python
@dataclass
class EmbeddingFingerprint:
    provider: str      # "ollama", "lmstudio"
    model: str         # "nomic-embed-text"
    dimension: int     # 768
    normalize: bool    # True
    version: str       # "1.0"
```

**How it works:**
1. **On ingestion:** Each chunk gets an `embedding_fingerprint` metadata field (from `ChunkMetadata.embedding_fingerprint`).
2. **On startup:** `verify_embedding_fingerprint(stored_fingerprint)` compares the stored fingerprint against the current config.
3. **On mismatch:** An `EmbeddingMismatchError` is raised (in `chroma_store.py`), alerting that re-indexing is needed.

```python
def verify_embedding_fingerprint(stored_fingerprint: str) -> bool:
    current = get_embedding_fingerprint()
    return current == stored_fingerprint
```

**Rollback strategy:**
- The fingerprint includes `version: str = "1.0"` for explicit versioning.
- Since ChromaDB collections are per-client, you can re-index one client at a time.
- The `ChromaClientVectorStore` accepts `verify_fingerprint=False` to skip verification during migration.
- BM25 indices are model-independent (text-only), so they survive embedding model changes without re-indexing.
