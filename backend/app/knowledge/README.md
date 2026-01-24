# Knowledge Graph Module (Future Integration)

## Status: Prepared for Future Use

This module provides entity extraction and knowledge graph storage capabilities.
Currently disabled but ready to integrate with the RAG pipeline.

## Overview

The knowledge graph module enables:
- Named entity extraction from documents
- Relationship discovery between entities
- Multi-hop reasoning for complex queries
- Entity disambiguation during retrieval

## Module Structure

```
backend/app/knowledge/
├── __init__.py           # Module exports
├── entity_extractor.py   # LLM-based entity extraction
├── graph_store.py        # SQLite-based graph storage
├── graph_query.py        # Graph traversal and querying
└── README.md             # This file
```

## How to Enable

1. Set `RAG__KNOWLEDGE_GRAPH_ENABLED=true` in `.env`
2. Entity extraction runs during document upload
3. Graph queries augment RAG retrieval

## Entity Types Supported

The entity extractor can identify:
- **PERSON** - Names of people
- **ORGANIZATION** - Company names, institutions
- **LOCATION** - Places, addresses
- **DATE** - Dates and time references
- **MONEY** - Monetary amounts
- **PERCENTAGE** - Percentage values
- **PRODUCT** - Product names, services
- **EVENT** - Named events

Custom entity types can be added via configuration.

## Integration Points

### 1. Document Processing (`processors/chunking.py`)

Chunks can store detected entities in metadata:

```python
ChunkMetadata(
    detected_entities=[
        {"type": "PERSON", "value": "John Smith"},
        {"type": "ORGANIZATION", "value": "Acme Corp"},
    ],
    entity_ids=["entity_123", "entity_456"],  # Links to KG nodes
    semantic_type="fact",  # fact, definition, procedure, example
)
```

### 2. Retrieval Agent (`agents/retrieval_agent.py`)

When enabled, graph expansion can augment retrieval:

```python
# Expand query with related entities
related_entities = await graph_query.expand(query_entities, depth=2)

# Include related documents in retrieval
expanded_results = await retrieve_with_entities(query, related_entities)
```

### 3. Document Upload Flow

```
Document Upload
     │
     ▼
Text Extraction (Docling)
     │
     ▼
Chunking with Metadata
     │
     ├── [KG Disabled] Direct to Vector Store
     │
     └── [KG Enabled]
            │
            ▼
        Entity Extraction
            │
            ▼
        Graph Storage
            │
            ▼
        Vector Store (with entity_ids in metadata)
```

## Database Schema

The knowledge graph uses SQLite with the following schema:

```sql
-- Entities (nodes)
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    value TEXT NOT NULL,
    client_id TEXT NOT NULL,
    metadata TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Relationships (edges)
CREATE TABLE relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    metadata TEXT,  -- JSON
    FOREIGN KEY (source_id) REFERENCES entities(id),
    FOREIGN KEY (target_id) REFERENCES entities(id)
);

-- Entity-to-chunk mapping
CREATE TABLE entity_chunks (
    entity_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (entity_id, chunk_id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);
```

## Usage Example (Future)

```python
from app.knowledge import get_entity_extractor, get_graph_store

# Extract entities from text
extractor = get_entity_extractor()
entities = await extractor.extract(chunk_text)

# Store in graph
graph = get_graph_store(client_id="client_123")
for entity in entities:
    graph.add_entity(entity)

# Query related entities
related = graph.get_related_entities(
    entity_id="entity_123",
    relation_types=["works_for", "located_in"],
    max_depth=2
)
```

## Configuration

```bash
# .env
RAG__KNOWLEDGE_GRAPH_ENABLED=false  # Set to true to enable
RAG__KG_EXPANSION_DEPTH=2           # Relationship hops to traverse
RAG__KG_PERSIST_PATH=./data/knowledge_graphs
```

## Future Enhancements

1. **Relationship Extraction** - Automatic relationship discovery from text
2. **Entity Resolution** - Merge duplicate entities across documents
3. **Graph Embeddings** - Embed graph structure for similarity search
4. **Temporal Reasoning** - Track entity changes over time
5. **Cross-Client Graphs** - Shared global knowledge graph
