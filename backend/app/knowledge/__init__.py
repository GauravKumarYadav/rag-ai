"""
Knowledge Graph module for per-client entity and relationship storage.

Provides:
- Entity extraction from documents
- Relationship detection between entities
- SQLite-based graph storage per client
- Query expansion using graph relationships
"""

from app.knowledge.graph_store import (
    KnowledgeGraphStore,
    Entity,
    Relationship,
    Mention,
    get_knowledge_graph,
)
from app.knowledge.entity_extractor import (
    EntityExtractor,
    get_entity_extractor,
)
from app.knowledge.graph_query import (
    GraphQueryExpander,
    get_graph_query_expander,
)

__all__ = [
    # Graph store
    "KnowledgeGraphStore",
    "Entity",
    "Relationship",
    "Mention",
    "get_knowledge_graph",
    
    # Entity extraction
    "EntityExtractor",
    "get_entity_extractor",
    
    # Query expansion
    "GraphQueryExpander",
    "get_graph_query_expander",
]
