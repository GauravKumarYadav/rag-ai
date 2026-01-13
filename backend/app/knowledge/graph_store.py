"""
Knowledge Graph Storage using SQLite.

Provides per-client isolated knowledge graphs storing:
- Entities (people, organizations, documents, dates, amounts)
- Relationships between entities
- Mentions linking entities to document chunks

SQLite is used for:
- Simplicity (no additional infrastructure)
- Per-client isolation (separate databases)
- ACID compliance
- Full-text search support
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


# Entity types
class EntityType:
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    DOCUMENT = "DOCUMENT"
    DATE = "DATE"
    AMOUNT = "AMOUNT"
    PRODUCT = "PRODUCT"
    LOCATION = "LOCATION"
    OTHER = "OTHER"


# Relationship types
class RelationType:
    AUTHORED = "AUTHORED"
    REFERENCES = "REFERENCES"
    WORKS_FOR = "WORKS_FOR"
    DATED = "DATED"
    OWNS = "OWNS"
    MENTIONS = "MENTIONS"
    RELATED_TO = "RELATED_TO"
    SIGNED = "SIGNED"
    PAID = "PAID"
    RECEIVED = "RECEIVED"


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    id: str
    name: str
    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    @classmethod
    def create(cls, name: str, type: str, **attributes) -> "Entity":
        """Create a new entity with generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            type=type,
            attributes=attributes,
            created_at=datetime.utcnow(),
        )


@dataclass
class Relationship:
    """A relationship between two entities."""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 1.0
    source_doc_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    @classmethod
    def create(
        cls,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float = 1.0,
        source_doc_id: Optional[str] = None,
        **attributes
    ) -> "Relationship":
        """Create a new relationship with generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            source_doc_id=source_doc_id,
            attributes=attributes,
            created_at=datetime.utcnow(),
        )


@dataclass
class Mention:
    """A mention linking an entity to a document chunk."""
    entity_id: str
    doc_chunk_id: str
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    context: Optional[str] = None


# SQL Schema
SCHEMA_SQL = """
-- Entities table
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    attributes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Index for entity lookups
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);

-- Relationships table
CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES entities(id),
    target_id TEXT NOT NULL REFERENCES entities(id),
    relation_type TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source_doc_id TEXT,
    attributes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for relationship queries
CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type);

-- Entity mentions table (links entities to document chunks)
CREATE TABLE IF NOT EXISTS mentions (
    entity_id TEXT NOT NULL REFERENCES entities(id),
    doc_chunk_id TEXT NOT NULL,
    span_start INTEGER,
    span_end INTEGER,
    context TEXT,
    PRIMARY KEY (entity_id, doc_chunk_id)
);

-- Index for chunk lookups
CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON mentions(doc_chunk_id);
"""


class KnowledgeGraphStore:
    """
    SQLite-based knowledge graph storage.
    
    Each client gets their own SQLite database for complete isolation.
    """
    
    def __init__(self, client_id: str, persist_path: Optional[str] = None):
        """
        Initialize knowledge graph for a client.
        
        Args:
            client_id: Unique client identifier
            persist_path: Directory to store SQLite databases
        """
        self.client_id = client_id
        self.persist_path = Path(persist_path or settings.rag.kg_persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.persist_path / f"kg_{client_id}.db"
        self._conn: Optional[sqlite3.Connection] = None
        
        # Initialize database
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database with schema."""
        conn = self._get_connection()
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.debug(f"Initialized knowledge graph DB for client {self.client_id}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    # ==========================================================================
    # Entity Operations
    # ==========================================================================
    
    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the graph."""
        conn = self._get_connection()
        conn.execute(
            """INSERT OR REPLACE INTO entities (id, name, type, attributes, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                entity.id,
                entity.name,
                entity.type,
                json.dumps(entity.attributes),
                entity.created_at.isoformat() if entity.created_at else None,
            )
        )
        conn.commit()
        return entity.id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        
        if row:
            return self._row_to_entity(row)
        return None
    
    def find_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """Find entities by name and/or type."""
        conn = self._get_connection()
        
        query = "SELECT * FROM entities WHERE 1=1"
        params: List[Any] = []
        
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        
        if entity_type:
            query += " AND type = ?"
            params.append(entity_type)
        
        query += f" LIMIT {limit}"
        
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_entity(row) for row in rows]
    
    def find_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> Optional[Entity]:
        """Find an entity by exact name match."""
        conn = self._get_connection()
        
        if entity_type:
            row = conn.execute(
                "SELECT * FROM entities WHERE name = ? AND type = ?",
                (name, entity_type)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM entities WHERE name = ?", (name,)
            ).fetchone()
        
        if row:
            return self._row_to_entity(row)
        return None
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships."""
        conn = self._get_connection()
        
        # Delete relationships
        conn.execute("DELETE FROM relationships WHERE source_id = ? OR target_id = ?", 
                    (entity_id, entity_id))
        # Delete mentions
        conn.execute("DELETE FROM mentions WHERE entity_id = ?", (entity_id,))
        # Delete entity
        cursor = conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        conn.commit()
        
        return cursor.rowcount > 0
    
    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        """Convert a database row to Entity object."""
        return Entity(
            id=row["id"],
            name=row["name"],
            type=row["type"],
            attributes=json.loads(row["attributes"]) if row["attributes"] else {},
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )
    
    # ==========================================================================
    # Relationship Operations
    # ==========================================================================
    
    def add_relationship(self, rel: Relationship) -> str:
        """Add a relationship to the graph."""
        conn = self._get_connection()
        conn.execute(
            """INSERT OR REPLACE INTO relationships 
               (id, source_id, target_id, relation_type, confidence, source_doc_id, attributes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rel.id,
                rel.source_id,
                rel.target_id,
                rel.relation_type,
                rel.confidence,
                rel.source_doc_id,
                json.dumps(rel.attributes),
                rel.created_at.isoformat() if rel.created_at else None,
            )
        )
        conn.commit()
        return rel.id
    
    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: Optional[str] = None,
    ) -> List[Tuple[Relationship, Entity]]:
        """
        Get relationships for an entity.
        
        Args:
            entity_id: The entity to query
            direction: "outgoing", "incoming", or "both"
            relation_type: Filter by relationship type
            
        Returns:
            List of (relationship, related_entity) tuples
        """
        conn = self._get_connection()
        results: List[Tuple[Relationship, Entity]] = []
        
        # Build queries based on direction
        queries = []
        
        if direction in ("outgoing", "both"):
            q = """SELECT r.*, e.* FROM relationships r 
                   JOIN entities e ON r.target_id = e.id 
                   WHERE r.source_id = ?"""
            if relation_type:
                q += " AND r.relation_type = ?"
            queries.append((q, "target"))
        
        if direction in ("incoming", "both"):
            q = """SELECT r.*, e.* FROM relationships r 
                   JOIN entities e ON r.source_id = e.id 
                   WHERE r.target_id = ?"""
            if relation_type:
                q += " AND r.relation_type = ?"
            queries.append((q, "source"))
        
        for query, _ in queries:
            params: List[Any] = [entity_id]
            if relation_type:
                params.append(relation_type)
            
            rows = conn.execute(query, params).fetchall()
            for row in rows:
                rel = Relationship(
                    id=row["id"],
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    relation_type=row["relation_type"],
                    confidence=row["confidence"],
                    source_doc_id=row["source_doc_id"],
                    attributes=json.loads(row["attributes"]) if row["attributes"] else {},
                )
                entity = self._row_to_entity(row)
                results.append((rel, entity))
        
        return results
    
    def get_related_entities(
        self,
        entity_id: str,
        depth: int = 1,
        relation_types: Optional[List[str]] = None,
    ) -> List[Entity]:
        """
        Get entities related to the given entity within N hops.
        
        Args:
            entity_id: Starting entity
            depth: Maximum relationship hops
            relation_types: Filter by relationship types
            
        Returns:
            List of related entities
        """
        visited: Set[str] = {entity_id}
        current_level = {entity_id}
        related: List[Entity] = []
        
        for _ in range(depth):
            next_level: Set[str] = set()
            
            for eid in current_level:
                relationships = self.get_relationships(
                    eid, 
                    direction="both",
                    relation_type=relation_types[0] if relation_types and len(relation_types) == 1 else None
                )
                
                for rel, entity in relationships:
                    if relation_types and rel.relation_type not in relation_types:
                        continue
                    
                    if entity.id not in visited:
                        visited.add(entity.id)
                        next_level.add(entity.id)
                        related.append(entity)
            
            current_level = next_level
            if not current_level:
                break
        
        return related
    
    # ==========================================================================
    # Mention Operations
    # ==========================================================================
    
    def add_mention(self, mention: Mention) -> None:
        """Add an entity mention in a document chunk."""
        conn = self._get_connection()
        conn.execute(
            """INSERT OR REPLACE INTO mentions 
               (entity_id, doc_chunk_id, span_start, span_end, context)
               VALUES (?, ?, ?, ?, ?)""",
            (
                mention.entity_id,
                mention.doc_chunk_id,
                mention.span_start,
                mention.span_end,
                mention.context,
            )
        )
        conn.commit()
    
    def get_entities_for_chunk(self, doc_chunk_id: str) -> List[Entity]:
        """Get all entities mentioned in a document chunk."""
        conn = self._get_connection()
        rows = conn.execute(
            """SELECT e.* FROM entities e
               JOIN mentions m ON e.id = m.entity_id
               WHERE m.doc_chunk_id = ?""",
            (doc_chunk_id,)
        ).fetchall()
        
        return [self._row_to_entity(row) for row in rows]
    
    def get_chunks_for_entity(self, entity_id: str) -> List[str]:
        """Get all document chunks that mention an entity."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT doc_chunk_id FROM mentions WHERE entity_id = ?",
            (entity_id,)
        ).fetchall()
        
        return [row["doc_chunk_id"] for row in rows]
    
    # ==========================================================================
    # Statistics
    # ==========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        conn = self._get_connection()
        
        entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        mention_count = conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
        
        # Entity type distribution
        type_dist = conn.execute(
            "SELECT type, COUNT(*) as count FROM entities GROUP BY type"
        ).fetchall()
        
        return {
            "client_id": self.client_id,
            "entity_count": entity_count,
            "relationship_count": rel_count,
            "mention_count": mention_count,
            "entity_types": {row["type"]: row["count"] for row in type_dist},
        }


# Cache of knowledge graphs per client
_kg_cache: Dict[str, KnowledgeGraphStore] = {}


def get_knowledge_graph(client_id: str) -> KnowledgeGraphStore:
    """Get or create a knowledge graph for a client."""
    if client_id not in _kg_cache:
        _kg_cache[client_id] = KnowledgeGraphStore(client_id)
    return _kg_cache[client_id]


def clear_kg_cache() -> None:
    """Clear the knowledge graph cache."""
    for kg in _kg_cache.values():
        kg.close()
    _kg_cache.clear()
