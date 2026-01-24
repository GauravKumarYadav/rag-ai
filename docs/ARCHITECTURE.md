# Architecture Overview

This document describes the architecture of the Agentic RAG Chatbot.

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Docker Compose                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐│
│  │    Redis     │  │   ChromaDB   │  │      Chat-API         ││
│  │              │  │              │  │  ┌─────────────────┐  ││
│  │  Sessions    │  │  Vector      │  │  │   LangGraph     │  ││
│  │  Memory      │  │  Store       │  │  │   Orchestrator  │  ││
│  │  Summaries   │  │  (Global +   │  │  └────────┬────────┘  ││
│  │              │  │   Client)    │  │           │           ││
│  └──────────────┘  └──────────────┘  │  ┌────────┴────────┐  ││
│         │                   │        │  │                 │  ││
│         └───────────────────┴────────┤  │  Query Agent    │  ││
│                                      │  │  Retrieval Agent│  ││
│                                      │  │  Tool Agent     │  ││
│                                      │  │  Synthesis Agent│  ││
│                                      │  └─────────────────┘  ││
│                                      └───────────────────────┘│
├────────────────────────────────────────────────────────────────┤
│                   Nginx (Frontend + Proxy)                     │
└────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ LM Studio│   │LangSmith │   │  Users   │
        │ (LLM)    │   │(Tracing) │   │          │
        └──────────┘   └──────────┘   └──────────┘
```

## Multi-Agent System

### Agent Overview

| Agent | Responsibility |
|-------|----------------|
| **Orchestrator** | Coordinates agent workflow via LangGraph |
| **Query Agent** | Intent classification, query rewriting |
| **Retrieval Agent** | RAG from client + global collections |
| **Tool Agent** | Calculator, DateTime tools |
| **Synthesis Agent** | Response generation with sources |

### Agent State

All agents share a common state (`AgentState`) that flows through the graph:

```python
class AgentState(TypedDict):
    # Input
    message: str
    client_id: str
    conversation_id: str
    
    # Memory
    conversation_summary: str
    recent_messages: List[Dict]
    
    # Query Processing
    intent: str  # chitchat, question, follow_up, tool
    needs_retrieval: bool
    rewritten_query: str
    
    # Retrieval
    retrieved_chunks: List[RetrievalHit]
    
    # Tools
    tool_name: Optional[str]
    tool_result: Optional[str]
    
    # Output
    response: str
    sources: List[Dict]
```

### Workflow Graph

```
┌─────────┐
│  Start  │
└────┬────┘
     │
     ▼
┌─────────────┐
│ Query Agent │ ─── Classify intent, detect tools, rewrite query
└──────┬──────┘
       │
       ├──[intent=chitchat]────────────────────┐
       │                                       │
       ├──[tool_name set]──┐                   │
       │                   ▼                   │
       │           ┌─────────────┐             │
       │           │ Tool Agent  │             │
       │           └──────┬──────┘             │
       │                  │                    │
       ├──[needs_retrieval]                    │
       │                   │                   │
       ▼                   │                   │
┌───────────────┐          │                   │
│ Retrieval     │          │                   │
│ Agent         │          │                   │
└───────┬───────┘          │                   │
        │                  │                   │
        └──────────────────┼───────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Synthesis   │
                    │ Agent       │
                    └──────┬──────┘
                           │
                           ▼
                       ┌───────┐
                       │  END  │
                       └───────┘
```

## Memory System

### Auto-Summarization Flow

```
User Message
     │
     ▼
┌─────────────────────────────────────┐
│ Add to Session Buffer (Redis)       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ Check: Total tokens > threshold?    │
└─────────────────┬───────────────────┘
                  │
        ┌─────────┴─────────┐
        │ No                │ Yes
        ▼                   ▼
   Continue         ┌───────────────────┐
                    │ Summarize older   │
                    │ messages via LLM  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Store summary in  │
                    │ Redis             │
                    └───────────────────┘
```

### Memory Storage

| Data | Storage | TTL |
|------|---------|-----|
| Session messages | Redis list | 24h |
| Conversation summary | Redis string | 24h |
| Session metadata | Redis hash | 24h |

## RAG System

### Collection Strategy

- **Global Collection** (`global_docs`): Documents available to all clients
- **Client Collection** (`client_{id}_docs`): Client-specific documents

### Retrieval Flow

```
Query
  │
  ▼
┌─────────────────────────────────────┐
│ Search Client Collection            │
│ (BM25 + Vector → RRF Fusion)        │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ Search Global Collection            │
│ (BM25 + Vector → RRF Fusion)        │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ Merge & Deduplicate Results         │
│ (Client results have priority)      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ Rerank with Cross-Encoder           │
│ (Optional, enabled by default)      │
└─────────────────┬───────────────────┘
                  │
                  ▼
         Top K Results
```

## Document Processing

### Processing Pipeline

```
Document Upload
       │
       ▼
┌─────────────────────────────────────┐
│ Docling Processor                   │
│ (PDF, DOCX, Images → Markdown)      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ Chunking (with rich metadata)       │
│ - Respects sentence boundaries      │
│ - Respects heading boundaries       │
│ - Content-hash IDs for deduplication│
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ ChromaDB Storage                    │
│ (Embeddings generated via LM Studio)│
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ BM25 Index Update                   │
│ (Keyword search index)              │
└─────────────────────────────────────┘
```

### Chunk Metadata

```python
class ChunkMetadata:
    # Core fields
    doc_id: str
    client_id: str
    chunk_index: int
    source_filename: str
    
    # Position info
    page_number: Optional[int]
    section_heading: Optional[str]
    
    # KG-ready fields (future)
    detected_entities: List[Dict]
    semantic_type: Optional[str]  # fact, definition, procedure
```

## Data Persistence

All data is persisted via Docker named volumes:

| Data | Volume | Service |
|------|--------|---------|
| Vector embeddings | `rag-chroma-data` | ChromaDB |
| Sessions & memory | `rag-redis-data` | Redis |
| BM25 index | `./data/bm25` (bind) | Chat-API |

This ensures data survives container restarts and redeployments.

## Security

### Authentication

- JWT-based authentication
- Configurable token expiration
- Superuser role for admin endpoints

### Client Isolation

- Each client has isolated document collections
- Client ID validated on all document operations
- Global collection for shared documents

## Future: Knowledge Graph Integration

The system is prepared for future knowledge graph integration:

1. **Chunk metadata includes KG fields**: `detected_entities`, `entity_ids`, `semantic_type`
2. **Knowledge module documented**: `backend/app/knowledge/README.md`
3. **Entity extraction ready**: Uses LM Studio for NER
4. **SQLite graph storage**: Scalable to Neo4j later

Enable with: `RAG__KNOWLEDGE_GRAPH_ENABLED=true`
