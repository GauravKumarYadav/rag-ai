# Architecture Overview

This document describes the overall system architecture, key modules, and data flow for the RAG AI Chatbot.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Frontend (Vanilla HTML/JS)                      │
│   index.html (Chat UI)  │  admin.html (Admin Dashboard)                     │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ HTTP / WebSocket
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             FastAPI Backend                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   Auth   │  │   Chat   │  │Documents │  │  Admin   │  │Evaluation│      │
│  │  Routes  │  │  Routes  │  │  Routes  │  │  Routes  │  │  Routes  │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                       Middleware Layer                              │    │
│  │  CORS │ Audit Logging │ Correlation ID │ Prometheus Metrics        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                   Optimized RAG Pipeline                            │    │
│  │  QueryProcessor │ Reranker │ Compressor │ EvidenceValidator        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                      Core Services                                  │    │
│  │  ChatService │ PromptBuilder │ StateManager │ TokenBudgeter        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              ▼                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  LLM Client │  │ RAG/Vector  │  │  Memory     │  │ Evaluation  │        │
│  │  (LMStudio) │  │  Store      │  │  (Long-term)│  │  Framework  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
┌──────────────┐          ┌──────────────┐              ┌──────────────┐
│   ChromaDB   │          │    MySQL     │              │   LMStudio   │
│ (Embeddings) │          │ (Audit Logs) │              │   (LLM API)  │
└──────────────┘          └──────────────┘              └──────────────┘
```

---

## Small Model RAG Optimization

The system includes an optimized RAG pipeline specifically designed for smaller language models:

### Optimization Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Small Model RAG Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. STATE MANAGEMENT                                                        │
│     ┌─────────────────┐                                                     │
│     │ ConversationState│ → user_goal, current_task, entities,              │
│     │ (Redis-backed)   │   constraints, decisions, open_questions          │
│     └─────────────────┘                                                     │
│              ↓                                                              │
│  2. INTENT CLASSIFICATION & QUERY REWRITING                                 │
│     ┌─────────────────┐                                                     │
│     │ QueryProcessor  │ → chitchat | question | follow_up | action         │
│     │                 │ → Reference resolution ("that doc", etc.)          │
│     │                 │ → Query rewriting for optimal retrieval            │
│     └─────────────────┘                                                     │
│              ↓                                                              │
│  3. HYBRID RETRIEVAL (BM25 + Vector → RRF → Rerank → MMR)                  │
│     ┌─────────────────┐                                                     │
│     │ HybridSearch    │ → BM25 keyword search (rank-bm25)                  │
│     │                 │ → Vector search (ChromaDB)                         │
│     │                 │ → Reciprocal Rank Fusion (RRF)                     │
│     │ + Reranker      │ → Cross-encoder reranking (sentence-transformers) │
│     │ + MMR           │ → Diversity via Maximal Marginal Relevance        │
│     └─────────────────┘                                                     │
│              ↓                                                              │
│  3b. KNOWLEDGE GRAPH EXPANSION (optional, per-client)                       │
│     ┌─────────────────┐                                                     │
│     │ GraphQuery      │ → Entity extraction from query                     │
│     │                 │ → Graph traversal (1-2 hops)                       │
│     │                 │ → Related entity/chunk retrieval                   │
│     └─────────────────┘                                                     │
│              ↓                                                              │
│  4. COMPRESSION (raw chunks → dense bullets)                                │
│     ┌─────────────────┐                                                     │
│     │ ContextCompressor│ → Extractive: TF-IDF key sentence extraction     │
│     │                 │ → LLM refinement: atomic bullets with citations   │
│     └─────────────────┘                                                     │
│              ↓                                                              │
│  5. EVIDENCE VALIDATION                                                     │
│     ┌─────────────────┐                                                     │
│     │ EvidenceValidator│ → Confidence scoring                              │
│     │                 │ → Contradiction detection                          │
│     │                 │ → Disclaimer generation if low confidence          │
│     └─────────────────┘                                                     │
│              ↓                                                              │
│  6. TOKEN BUDGETING (hard cap ~1000 tokens)                                 │
│     ┌─────────────────┐                                                     │
│     │ TokenBudgeter   │ → Priority-based fact selection                    │
│     │                 │ → Sentence truncation for oversized facts          │
│     └─────────────────┘                                                     │
│              ↓                                                              │
│  7. PROMPT BUILDING                                                         │
│     ┌─────────────────┐                                                     │
│     │ PromptBuilder   │ → State block (5-15 lines)                         │
│     │                 │ → Compressed facts with citations                  │
│     │                 │ → Sliding window (last 3 turns)                    │
│     │                 │ → Running summary + episodic memories              │
│     │                 │ → Citation enforcement instructions                │
│     └─────────────────┘                                                     │
│              ↓                                                              │
│  8. ANSWER VERIFICATION (post-generation)                                   │
│     ┌─────────────────┐                                                     │
│     │ AnswerVerifier  │ → Extract claims from response                     │
│     │                 │ → Cross-check against retrieved context            │
│     │                 │ → Flag unsupported claims                          │
│     │ CitationExtractor│ → Validate citation coverage                      │
│     │                 │ → Warn if below threshold (70%)                    │
│     └─────────────────┘                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **ConversationState** | `memory/conversation_state.py` | Structured state instead of raw history |
| **QueryProcessor** | `services/query_processor.py` | Intent classification + query rewriting |
| **HybridSearch** | `rag/hybrid_search.py` | BM25 + Vector fusion with RRF |
| **BM25Index** | `rag/bm25_index.py` | Keyword-based BM25 retrieval |
| **Reranker** | `rag/reranker.py` | Cross-encoder reranking with MMR |
| **ContextCompressor** | `services/context_compressor.py` | Hybrid extractive + LLM compression |
| **EvidenceValidator** | `services/evidence_validator.py` | Confidence scoring + contradiction detection |
| **TokenBudgeter** | `services/token_budgeter.py` | Hard token cap enforcement |
| **EntityExtractor** | `knowledge/entity_extractor.py` | LLM-based entity/relationship extraction |
| **GraphStore** | `knowledge/graph_store.py` | Per-client SQLite knowledge graphs |
| **GraphQuery** | `knowledge/graph_query.py` | KG traversal for query expansion |
| **CitationExtractor** | `services/citation_extractor.py` | Extract and validate citations |
| **AnswerVerifier** | `services/answer_verifier.py` | Ground truth verification of responses |
| **QualityMetrics** | `evaluation/quality_metrics.py` | Enhanced RAG evaluation metrics |

### Configuration

```bash
# .env settings for small model optimization

# Reranking
RAG__RERANKER_ENABLED=true
RAG__RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG__INITIAL_FETCH_K=30
RAG__RERANK_TOP_K=5
RAG__MMR_LAMBDA=0.5
RAG__CONTEXT_TOKEN_BUDGET=1000
RAG__MIN_CONFIDENCE_THRESHOLD=0.3

# Hybrid Search (BM25 + Vector)
RAG__BM25_ENABLED=true
RAG__BM25_WEIGHT=0.4
RAG__VECTOR_WEIGHT=0.6
RAG__BM25_PERSIST_PATH=./data/bm25

# Knowledge Graph
RAG__KNOWLEDGE_GRAPH_ENABLED=true
RAG__KG_EXPANSION_DEPTH=2
RAG__KG_PERSIST_PATH=./data/knowledge_graphs
RAG__ENTITY_EXTRACTION_MODEL=mistral:latest

# Answer Verification & Citation
RAG__VERIFICATION_ENABLED=true
RAG__CITATION_REQUIRED=true
RAG__MIN_CITATION_COVERAGE=0.7

# Session Memory
SESSION__SLIDING_WINDOW_TURNS=3
SESSION__EPISODIC_MEMORY_ENABLED=true
SESSION__RUNNING_SUMMARY_ENABLED=true
```

---

## Module Breakdown

### Authentication (`backend/app/auth/`)
- **JWT-based authentication** with bcrypt password hashing
- `dependencies.py` - `get_current_user`, `require_superuser`
- `jwt.py` - Token creation/validation
- `password.py` - Password hashing utilities

### API Routes (`backend/app/api/routes/`)
| Route File | Prefix | Description |
|------------|--------|-------------|
| `auth.py` | `/auth` | Login, logout, user info |
| `chat.py` | `/chat` | Chat completions (REST + streaming) |
| `websocket.py` | `/chat/ws` | WebSocket chat interface |
| `documents.py` | `/documents` | Upload, search, stats |
| `conversations.py` | `/conversations` | History, memory |
| `clients.py` | `/clients` | Multi-client management |
| `admin.py` | `/admin` | Audit logs, users, stats, config |
| `evaluation.py` | `/evaluation` | RAG evaluation datasets & runs |
| `health.py` | `/health` | Health checks |
| `status.py` | `/status` | System status |
| `models.py` | `/models` | LLM model info & switching |

### Core (`backend/app/core/`)
- `logging.py` - Structured JSON logging with correlation IDs, file rotation
- `metrics.py` - Prometheus instrumentation with custom RAG/LLM histograms

### RAG Pipeline (`backend/app/rag/`)
- `vector_store.py` - ChromaDB integration for documents & memories
- `retriever.py` - Semantic search with **reranking and MMR support**
- `reranker.py` - **Cross-encoder reranking and MMR diversity**
- `bm25_index.py` - **BM25 keyword index for hybrid search**
- `hybrid_search.py` - **BM25 + Vector fusion with Reciprocal Rank Fusion**
- `embeddings.py` - LMStudio embedding function
- `factory.py` - Vector store factory (supports ChromaDB, Pinecone, etc.)

### Knowledge Graph (`backend/app/knowledge/`)
- `entity_extractor.py` - **LLM-based entity and relationship extraction**
- `graph_store.py` - **Per-client SQLite graph storage**
- `graph_query.py` - **Query expansion via graph traversal**

### Memory (`backend/app/memory/`)
- `session_buffer.py` - **Sliding window conversation buffers with running summaries**
- `long_term.py` - **Summarization, episodic memory extraction, and long-term storage**
- `conversation_state.py` - **Structured conversation state management**
- `pruner.py` - Background cleanup scheduler

### Services (`backend/app/services/`)
- `chat_service.py` - **Optimized chat pipeline with all RAG optimizations**
- `query_processor.py` - **Intent classification and query rewriting**
- `context_compressor.py` - **Hybrid extractive + LLM context compression**
- `evidence_validator.py` - **Confidence scoring and contradiction detection**
- `token_budgeter.py` - **Token budget enforcement**
- `prompt_builder.py` - **Optimized prompt building with state blocks and citation enforcement**
- `citation_extractor.py` - **Extract and validate source citations from responses**
- `answer_verifier.py` - **Cross-check LLM answers against retrieved context**
- `client_extractor.py` - Client name extraction from messages

### Evaluation (`backend/app/evaluation/`)
- `generator.py` - Auto-generate Q&A pairs from documents using LLM
- `metrics.py` - Precision@K, Recall@K, MRR, Faithfulness
- `quality_metrics.py` - **Enhanced metrics: Hit Rate, Citation Accuracy, Answer Relevance**
- `runner.py` - Execute evaluations and store results
- `scheduler.py` - APScheduler for cron-based daily runs

### Database (`backend/app/db/`)
- `mysql.py` - Connection pool, audit logging, user queries

### Document Processing (`backend/app/processors/`)
- `registry.py` - Plugin registry for document processors
- Supports PDF, DOCX, TXT, images with OCR

---

## Data Flow

### Optimized Chat Request Flow (Small Model Pipeline)
```
1. User sends message (REST or WebSocket)
2. Auth middleware validates JWT
3. Correlation ID middleware adds request tracing
4. ChatService receives request
5. Load conversation state from Redis
6. QueryProcessor classifies intent and rewrites query
7. If retrieval needed:
   a. HybridSearch: BM25 + Vector → RRF fusion
   b. Knowledge Graph expansion (if enabled)
   c. Cross-encoder reranks candidates
   d. MMR selects diverse top-5
   e. ContextCompressor creates dense bullets
   f. EvidenceValidator checks confidence
   g. TokenBudgeter enforces budget
8. PromptBuilder creates optimized prompt with:
   - State block (5-15 lines)
   - Compressed facts with citations
   - Sliding window (last 3 turns)
   - Running summary + episodic memories
   - Citation enforcement instructions
9. LLM client streams response
10. Answer Verification (if enabled):
    - CitationExtractor validates source references
    - AnswerVerifier checks claims against context
    - Warnings appended for unsupported claims
11. Update conversation state and memory
12. Response streamed to user
```

### Legacy Chat Request Flow
```
1. User sends message (REST or WebSocket)
2. Auth middleware validates JWT
3. Correlation ID middleware adds request tracing
4. ChatService receives request
5. PromptBuilder constructs system prompt + context
6. RAG retriever fetches relevant document chunks
7. LLM client streams response
8. Session buffer stores conversation
9. Audit middleware logs request
10. Response streamed to user
```

### Document Upload Flow
```
1. User uploads file(s) with client_name
2. ProcessorRegistry selects appropriate processor
3. Text extracted (with OCR if needed)
4. Text chunked with overlap
5. Chunks embedded via LMStudio
6. Chunks stored in ChromaDB with metadata
7. Response returned with chunk count
```

### Evaluation Flow
```
1. Admin generates dataset from documents
2. LLM creates Q&A pairs from random chunks
3. Dataset stored in MySQL
4. Admin triggers evaluation run
5. For each Q&A: retrieve docs, compute metrics
6. Aggregated results stored in MySQL
7. Results viewable in admin dashboard
```

---

## Configuration

Settings are managed via `backend/app/config.py` with nested Pydantic models:

| Group | Key Settings |
|-------|--------------|
| `llm` | provider, model, temperature, timeout, context_window |
| `rag` | provider, chroma_db_path, embedding_model, **reranker settings**, **context budget** |
| `session` | max_tokens, max_messages, **sliding_window_turns**, **episodic_memory** |
| `mysql` | host, port, database, user, password |
| `redis` | url, session_ttl, cache_ttl |
| `logging` | level, log_dir, json_format, retention |
| `evaluation` | sample_size, cron_schedule, timezone |
| `jwt` | secret_key, algorithm, expire_minutes |

---

## Observability Stack

| Component | Purpose | Port |
|-----------|---------|------|
| Prometheus | Metrics collection | 9090 |
| Loki | Log aggregation | 3100 |
| Promtail | Log shipping | - |
| Grafana | Dashboards | 3001 |

Start with:
```bash
cd docker/monitoring
podman-compose up -d
```

---

## Security

- JWT tokens for API authentication
- bcrypt password hashing (12 rounds)
- Superuser role for admin endpoints
- Audit logging of all API requests
- CORS configured for allowed origins
- Credentials stored in environment variables
