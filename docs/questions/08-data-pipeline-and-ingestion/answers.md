# 8 — Data Pipeline & Ingestion — Answers

> All answers below are grounded in the **rag-ai** codebase.

---

## Q46. How do you design an ingestion pipeline that keeps the knowledge base fresh?

Our ingestion pipeline (from `ARCHITECTURE.md` and `processors/`):

```
Document Upload (API: POST /documents/upload)
       │
       ▼
Docling Processor (docling_processor.py)
  └─ PDF, DOCX, Images → Markdown
       │
       ▼
Chunking (chunking.py)
  └─ Heading-aware, sentence-boundary respecting
  └─ Content-hash IDs for deduplication
       │
       ▼
ChromaDB Storage (chroma_store.py)
  └─ Embeddings via LM Studio/Ollama/Groq
       │
       ▼
BM25 Index Update (bm25_index.py)
  └─ Keyword search index
```

**Freshness mechanisms:**

1. **Hot reload:** From `README.md`: *"Documents immediately available after upload"* — no server restart or batch job required.

2. **Content-hash IDs (`use_content_hash_ids = True`):** When a document is re-uploaded, chunks with identical content get the same hash ID. ChromaDB’s `upsert()` deduplicates automatically. Only changed/new chunks are added.

3. **Dual-index sync:** `HybridSearch.add_documents()` updates both BM25 and vector indices in a single call.

4. **Delete + re-add for updates:** Document updates are handled by deleting old chunks (by `doc_id`) and adding new ones. The `HybridSearch.delete()` method supports both ID-based and metadata-filter-based deletion.

5. **Per-client isolation:** Each client has separate collections, so updating one client’s documents doesn’t affect others.

6. **Ingestion script:** `scripts/ingest_documents.py` (3.0 KB) provides batch ingestion for bulk uploads.

---

## Q47. What pre-processing steps do you perform before embedding documents?

**Our pre-processing pipeline:**

**1. Format conversion (Docling — `docling_processor.py`):**
- PDF → Markdown (with OCR support for scanned documents)
- DOCX/DOC → Markdown
- Images (PNG, JPG, TIFF, BMP, GIF, WEBP) → text via OCR
- Plain text (TXT, MD) → passed through with simple decoding
- Throttled with `DOCLING_NUM_THREADS = 4` to avoid CPU saturation

**2. Chunking (`chunking.py`):**
- **Heading-aware splitting:** `respect_headings = True` breaks at markdown `#` headers.
- **Sentence-boundary respect:** `respect_sentences = True` avoids mid-sentence cuts.
- **Overlap:** `chunk_overlap = 200` characters ensures context continuity across chunk boundaries.
- **Minimum size filter:** `min_chunk_tokens = 50` drops tiny, uninformative chunks.

**3. Metadata enrichment (`ChunkMetadata`):**
Each chunk gets rich metadata:
- `doc_id`, `client_id`, `source_filename`
- `chunk_index`, `start_offset`, `end_offset`
- `page_number`, `section_heading` (from Docling)
- `embedding_fingerprint` (for model versioning)
- KG-ready fields: `detected_entities`, `entity_ids`, `semantic_type`

**4. Content-hash ID generation:**
```python
# Deterministic IDs via content hashing
use_content_hash_ids: bool = True
```
IDs are derived from `hashlib` digest of chunk content, enabling idempotent upserts.

**Why each step matters:**
- Docling ensures consistent markdown regardless of input format.
- Heading-aware chunking keeps topically coherent chunks.
- Content hashing enables deduplication without tracking external state.
- Rich metadata powers metadata filtering and source citation.

---

## Q48. How do you handle multi-modal data (tables, images, charts inside PDFs)?

Our system uses **Docling** (`docling_processor.py`) which handles multi-modal content:

**1. PDF tables:**
- Docling extracts tables and converts them to Markdown table format.
- The chunker’s `respect_headings` ensures tables stay with their section context.

**2. Images in PDFs:**
- Docling supports OCR via `PdfPipelineOptions` and `AcceleratorOptions`.
- Scanned PDFs are OCR’d to extract text from images.
- Our supported image formats: PNG, JPG, JPEG, TIFF, BMP, GIF, WEBP.

**3. Charts:**
- Chart text (labels, legends, titles) is extracted via OCR.
- Chart interpretation (understanding trends/data points) is limited to what OCR can extract.

**4. LLM model support:**
- Our default LLM is `Qwen3-VL-30B-Instruct` — a **vision-language** model. The "-VL" indicates it can process visual inputs, enabling future image understanding capabilities.

**Limitations and future work:**
- Complex charts with embedded data would benefit from a dedicated chart-to-data extraction step.
- The knowledge graph module (`entity_extractor.py`) could extract structured data from tables.

---

## Q49. Describe your approach to document-level and passage-level deduplication.

**Document-level deduplication:**
1. **Content-hash chunk IDs:** `use_content_hash_ids = True` generates deterministic IDs from chunk content via `hashlib`. Re-uploading the same document produces identical chunk IDs.
2. **ChromaDB upsert:** `collection.upsert()` with hash-based IDs is idempotent — identical chunks are silently deduplicated.
3. **Source tracking:** `source_filename` metadata enables detection of duplicate uploads at the file level.

**Passage-level deduplication:**
1. **Content hashing:** Two chunks from different documents with identical text content get the same hash ID, preventing duplicate passages.
2. **Merge deduplication in retrieval:** `RetrievalAgent._merge_results()` uses `seen_ids = set()` to deduplicate across client and global collections:
   ```python
   for hit in client_results:
       if hit.id not in seen_ids:
           seen_ids.add(hit.id)
           merged.append(hit)
   ```
3. **Document list deduplication:** `_get_document_list()` uses `seen_sources = set()` to deduplicate source filenames.

**Why this matters for retrieval:**
- Without deduplication, the same passage could consume multiple top-K slots, reducing diversity.
- The MMR selector (`reranker.py`) provides additional diversity by penalizing chunks too similar to already-selected ones.

---

## Q50. How do you detect and handle data drift in the knowledge base?

Data drift in a knowledge base manifests as:
- Outdated documents that no longer reflect current reality.
- Shifts in terminology or domain language that reduce retrieval effectiveness.
- New document types that the chunking strategy doesn’t handle well.

**Our detection mechanisms:**

1. **Embedding fingerprinting:** `EmbeddingFingerprint` detects when the embedding model changes, which could cause drift between old and new embeddings.

2. **Retrieval quality monitoring:** `HybridSearchResult` stats (`bm25_candidates`, `vector_candidates`, `fused_candidates`) track retrieval health over time. A drop in candidates suggests index drift.

3. **LangSmith tracing:** Monitoring answer quality trends across the `agentic-rag` project can surface degradation.

4. **Cost tracking trends:** If `prompt_tokens` are increasing over time for similar queries, it may indicate that the summarizer is struggling with drifted conversation patterns.

**Handling strategies:**

1. **Document metadata timestamps:** `uploaded_at` / `created_at` in chunk metadata enables recency-based filtering.
2. **Re-ingestion:** The hot-reload pipeline with content-hash IDs makes re-ingestion idempotent — you can periodically re-ingest the knowledge base to refresh it.
3. **Per-client isolation:** Drift in one client’s documents doesn’t affect others.
4. **BM25 index rebuild:** `BM25Index` supports full rebuild from persisted documents, recalibrating term frequencies for the current corpus.
