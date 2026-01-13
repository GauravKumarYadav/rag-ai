"""
Graph Query Expander - Expand retrieval queries using knowledge graph.

Uses the knowledge graph to:
1. Identify entities in the query
2. Find related entities via graph traversal
3. Expand the context with related document chunks
4. Provide entity disambiguation

This enables multi-hop reasoning and better entity resolution.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from app.clients.lmstudio import LMStudioClient, get_lmstudio_client
from app.config import settings
from app.knowledge.graph_store import (
    Entity,
    KnowledgeGraphStore,
    get_knowledge_graph,
)
from app.models.schemas import RetrievalHit

logger = logging.getLogger(__name__)


@dataclass
class QueryExpansion:
    """Result of query expansion using knowledge graph."""
    original_query: str
    identified_entities: List[Entity]
    related_entities: List[Entity]
    expanded_chunk_ids: List[str]
    disambiguation_hints: List[str]


@dataclass
class EntityMatch:
    """A potential entity match in a query."""
    entity: Entity
    match_text: str
    confidence: float
    start_pos: int
    end_pos: int


class GraphQueryExpander:
    """
    Expand queries using the knowledge graph.
    
    Pipeline:
    1. Entity linking: Find entities mentioned in query
    2. Graph expansion: Get related entities via relationships
    3. Chunk expansion: Get document chunks for related entities
    4. Disambiguation: Provide hints for ambiguous references
    """
    
    def __init__(
        self,
        lm_client: Optional[LMStudioClient] = None,
        expansion_depth: int = 2,
    ):
        """
        Initialize expander.
        
        Args:
            lm_client: LLM client for entity linking (optional)
            expansion_depth: How many hops to traverse in graph
        """
        self.lm_client = lm_client
        self.expansion_depth = expansion_depth or settings.rag.kg_expansion_depth
    
    def expand(
        self,
        query: str,
        client_id: str,
        max_related: int = 10,
    ) -> QueryExpansion:
        """
        Expand a query using the knowledge graph.
        
        Args:
            query: The search query
            client_id: Client ID for the knowledge graph
            max_related: Maximum number of related entities to return
            
        Returns:
            QueryExpansion with entities and expanded chunks
        """
        kg = get_knowledge_graph(client_id)
        
        # Step 1: Identify entities in query
        identified = self._identify_entities(query, kg)
        logger.debug(f"Identified {len(identified)} entities in query")
        
        # Step 2: Get related entities via graph traversal
        related: List[Entity] = []
        seen_ids: Set[str] = {e.id for e in identified}
        
        for entity in identified:
            entity_related = kg.get_related_entities(
                entity.id,
                depth=self.expansion_depth,
            )
            for rel_entity in entity_related:
                if rel_entity.id not in seen_ids and len(related) < max_related:
                    related.append(rel_entity)
                    seen_ids.add(rel_entity.id)
        
        logger.debug(f"Found {len(related)} related entities")
        
        # Step 3: Get document chunks for all entities
        chunk_ids: Set[str] = set()
        for entity in identified + related:
            chunks = kg.get_chunks_for_entity(entity.id)
            chunk_ids.update(chunks)
        
        logger.debug(f"Expanded to {len(chunk_ids)} document chunks")
        
        # Step 4: Generate disambiguation hints
        hints = self._generate_disambiguation_hints(identified, related, kg)
        
        return QueryExpansion(
            original_query=query,
            identified_entities=identified,
            related_entities=related,
            expanded_chunk_ids=list(chunk_ids),
            disambiguation_hints=hints,
        )
    
    def _identify_entities(
        self,
        query: str,
        kg: KnowledgeGraphStore,
    ) -> List[Entity]:
        """
        Identify entities mentioned in the query.
        
        Uses a combination of:
        - Exact name matching
        - Fuzzy matching for partial names
        - Common word filtering
        """
        identified: List[Entity] = []
        query_lower = query.lower()
        
        # Get all entities from the graph
        all_entities = kg.find_entities(limit=1000)
        
        for entity in all_entities:
            name_lower = entity.name.lower()
            
            # Check for exact match or substring match
            if name_lower in query_lower or self._fuzzy_match(name_lower, query_lower):
                identified.append(entity)
        
        # Sort by name length (longer names = more specific = higher priority)
        identified.sort(key=lambda e: len(e.name), reverse=True)
        
        # Remove duplicates keeping most specific
        seen_types: Dict[str, Entity] = {}
        unique: List[Entity] = []
        for entity in identified:
            key = entity.type
            if key not in seen_types:
                seen_types[key] = entity
                unique.append(entity)
            elif len(entity.name) > len(seen_types[key].name):
                # Replace with more specific match
                unique.remove(seen_types[key])
                seen_types[key] = entity
                unique.append(entity)
        
        return unique[:10]  # Limit to top 10
    
    def _fuzzy_match(self, name: str, query: str, threshold: float = 0.8) -> bool:
        """Check if name fuzzy matches query."""
        # Simple word-based matching
        name_words = set(name.split())
        query_words = set(query.split())
        
        if not name_words:
            return False
        
        # Check if most name words appear in query
        matches = name_words.intersection(query_words)
        ratio = len(matches) / len(name_words)
        
        return ratio >= threshold
    
    def _generate_disambiguation_hints(
        self,
        identified: List[Entity],
        related: List[Entity],
        kg: KnowledgeGraphStore,
    ) -> List[str]:
        """Generate hints to disambiguate entities."""
        hints: List[str] = []
        
        # Group entities by name to find ambiguities
        name_groups: Dict[str, List[Entity]] = {}
        for entity in identified + related:
            name_lower = entity.name.lower()
            if name_lower not in name_groups:
                name_groups[name_lower] = []
            name_groups[name_lower].append(entity)
        
        # Generate hints for ambiguous names
        for name, entities in name_groups.items():
            if len(entities) > 1:
                types = [e.type for e in entities]
                hint = f"'{name}' could refer to: {', '.join(set(types))}"
                hints.append(hint)
        
        return hints
    
    def enrich_retrieval(
        self,
        query: str,
        hits: List[RetrievalHit],
        client_id: str,
    ) -> Tuple[List[RetrievalHit], QueryExpansion]:
        """
        Enrich retrieval results using knowledge graph.
        
        Args:
            query: The search query
            hits: Current retrieval hits
            client_id: Client ID
            
        Returns:
            Tuple of (enriched_hits, expansion_info)
        """
        expansion = self.expand(query, client_id)
        
        # Get current hit IDs
        current_ids = {hit.id for hit in hits}
        
        # Add expansion chunks that aren't already retrieved
        new_chunk_ids = [
            cid for cid in expansion.expanded_chunk_ids
            if cid not in current_ids
        ]
        
        # Note: To fully implement this, we'd need access to the vector store
        # to retrieve the actual content for new_chunk_ids.
        # For now, we just return the expansion info
        
        logger.info(
            f"Query expansion found {len(expansion.identified_entities)} entities, "
            f"{len(expansion.related_entities)} related, "
            f"{len(new_chunk_ids)} additional chunks"
        )
        
        return hits, expansion
    
    async def expand_with_llm(
        self,
        query: str,
        client_id: str,
    ) -> QueryExpansion:
        """
        Expand query using LLM for entity extraction.
        
        More accurate but slower than rule-based expansion.
        """
        if not self.lm_client:
            return self.expand(query, client_id)
        
        kg = get_knowledge_graph(client_id)
        
        # Use LLM to extract entity mentions from query
        prompt = f"""Extract entity names from this query:

Query: {query}

List only the entity names (people, organizations, documents, dates, amounts) mentioned.
Output one name per line, nothing else."""

        try:
            response = await self.lm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            
            # Parse entity names from response
            entity_names = [
                line.strip() 
                for line in response.strip().split('\n') 
                if line.strip()
            ]
            
            # Look up entities in KG
            identified: List[Entity] = []
            for name in entity_names:
                entity = kg.find_entity_by_name(name)
                if entity:
                    identified.append(entity)
                else:
                    # Try fuzzy search
                    matches = kg.find_entities(name=name, limit=1)
                    if matches:
                        identified.append(matches[0])
            
            # Continue with graph expansion
            related: List[Entity] = []
            seen_ids: Set[str] = {e.id for e in identified}
            
            for entity in identified:
                entity_related = kg.get_related_entities(
                    entity.id,
                    depth=self.expansion_depth,
                )
                for rel_entity in entity_related:
                    if rel_entity.id not in seen_ids:
                        related.append(rel_entity)
                        seen_ids.add(rel_entity.id)
            
            # Get chunks
            chunk_ids: Set[str] = set()
            for entity in identified + related:
                chunks = kg.get_chunks_for_entity(entity.id)
                chunk_ids.update(chunks)
            
            return QueryExpansion(
                original_query=query,
                identified_entities=identified,
                related_entities=related[:10],
                expanded_chunk_ids=list(chunk_ids),
                disambiguation_hints=[],
            )
            
        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")
            return self.expand(query, client_id)


# Factory functions
_expander_instance: Optional[GraphQueryExpander] = None


def get_graph_query_expander(
    lm_client: Optional[LMStudioClient] = None,
) -> GraphQueryExpander:
    """Get or create the graph query expander singleton."""
    global _expander_instance
    
    if _expander_instance is None:
        _expander_instance = GraphQueryExpander(
            lm_client=lm_client,
            expansion_depth=settings.rag.kg_expansion_depth,
        )
    
    return _expander_instance
