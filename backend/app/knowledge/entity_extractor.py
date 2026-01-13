"""
Entity Extractor - Extract entities and relationships from document chunks.

Uses LLM to identify:
- Named entities (people, organizations, locations)
- Document entities (contracts, reports, invoices)
- Numerical entities (amounts, dates, percentages)
- Relationships between entities
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.clients.lmstudio import LMStudioClient, get_lmstudio_client
from app.config import settings
from app.knowledge.graph_store import (
    Entity,
    EntityType,
    KnowledgeGraphStore,
    Mention,
    Relationship,
    RelationType,
    get_knowledge_graph,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of entity extraction from a document chunk."""
    entities: List[Entity]
    relationships: List[Relationship]
    mentions: List[Mention]
    raw_extraction: Dict[str, Any]


ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from this document chunk.

DOCUMENT CHUNK:
{chunk_text}

DOCUMENT SOURCE: {source_name}

Extract the following entity types:
- PERSON: Names of people
- ORG: Company, organization, institution names
- DOCUMENT: References to other documents (contracts, reports, etc.)
- DATE: Specific dates or date ranges
- AMOUNT: Monetary amounts, quantities, percentages
- PRODUCT: Products, services, or items mentioned
- LOCATION: Places, addresses

Also identify relationships between entities:
- AUTHORED: Person authored/created document
- WORKS_FOR: Person works for organization
- REFERENCES: Document references another document
- DATED: Entity has a date association
- OWNS: Entity owns something
- SIGNED: Person signed document
- PAID/RECEIVED: Financial transactions

Output ONLY valid JSON (no markdown, no explanation):
{{
  "entities": [
    {{"name": "exact name as in text", "type": "PERSON|ORG|DOCUMENT|DATE|AMOUNT|PRODUCT|LOCATION", "attributes": {{"role": "optional role", "value": "optional value"}}}}
  ],
  "relationships": [
    {{"source": "entity_name", "target": "entity_name", "type": "AUTHORED|WORKS_FOR|REFERENCES|DATED|OWNS|SIGNED|PAID|RECEIVED|RELATED_TO"}}
  ]
}}

Rules:
- Extract entities EXACTLY as they appear in the text
- Only include entities clearly present in the text
- Relationships must connect entities you extracted
- Use RELATED_TO for unclear relationships
- Keep attributes minimal and factual"""


class EntityExtractor:
    """
    Extract entities and relationships from document chunks using LLM.
    """
    
    def __init__(
        self,
        lm_client: LMStudioClient,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize extractor.
        
        Args:
            lm_client: LLM client for extraction
            confidence_threshold: Minimum confidence for entities
        """
        self.lm_client = lm_client
        self.confidence_threshold = confidence_threshold
    
    async def extract(
        self,
        chunk_text: str,
        chunk_id: str,
        source_name: str = "unknown",
    ) -> ExtractionResult:
        """
        Extract entities and relationships from a chunk.
        
        Args:
            chunk_text: The document chunk text
            chunk_id: Unique identifier for the chunk
            source_name: Name of the source document
            
        Returns:
            ExtractionResult with entities, relationships, and mentions
        """
        # Build prompt
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            chunk_text=chunk_text,
            source_name=source_name,
        )
        
        try:
            # Call LLM
            response = await self.lm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            
            # Parse response
            return self._parse_extraction(response, chunk_id)
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return ExtractionResult(
                entities=[],
                relationships=[],
                mentions=[],
                raw_extraction={"error": str(e)},
            )
    
    def _parse_extraction(
        self,
        response: str,
        chunk_id: str,
    ) -> ExtractionResult:
        """Parse the LLM extraction response."""
        # Extract JSON from response
        json_str = self._extract_json(response)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction JSON: {e}")
            return ExtractionResult(
                entities=[],
                relationships=[],
                mentions=[],
                raw_extraction={"error": str(e), "raw": response},
            )
        
        # Parse entities
        entities: List[Entity] = []
        entity_map: Dict[str, Entity] = {}  # name -> entity for relationship resolution
        
        for ent_data in data.get("entities", []):
            name = ent_data.get("name", "").strip()
            ent_type = ent_data.get("type", "OTHER").upper()
            
            if not name:
                continue
            
            # Normalize type
            ent_type = self._normalize_entity_type(ent_type)
            
            entity = Entity.create(
                name=name,
                type=ent_type,
                **(ent_data.get("attributes") or {})
            )
            entities.append(entity)
            entity_map[name.lower()] = entity
        
        # Parse relationships
        relationships: List[Relationship] = []
        
        for rel_data in data.get("relationships", []):
            source_name = rel_data.get("source", "").strip().lower()
            target_name = rel_data.get("target", "").strip().lower()
            rel_type = rel_data.get("type", "RELATED_TO").upper()
            
            # Resolve to entity IDs
            source_entity = entity_map.get(source_name)
            target_entity = entity_map.get(target_name)
            
            if source_entity and target_entity:
                rel = Relationship.create(
                    source_id=source_entity.id,
                    target_id=target_entity.id,
                    relation_type=self._normalize_relation_type(rel_type),
                    source_doc_id=chunk_id,
                )
                relationships.append(rel)
        
        # Create mentions for all entities
        mentions: List[Mention] = [
            Mention(entity_id=ent.id, doc_chunk_id=chunk_id)
            for ent in entities
        ]
        
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            mentions=mentions,
            raw_extraction=data,
        )
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain other content."""
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json_match.group()
        return text
    
    def _normalize_entity_type(self, ent_type: str) -> str:
        """Normalize entity type to standard values."""
        type_map = {
            "PERSON": EntityType.PERSON,
            "PEOPLE": EntityType.PERSON,
            "NAME": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "COMPANY": EntityType.ORGANIZATION,
            "DOCUMENT": EntityType.DOCUMENT,
            "DOC": EntityType.DOCUMENT,
            "FILE": EntityType.DOCUMENT,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
            "AMOUNT": EntityType.AMOUNT,
            "MONEY": EntityType.AMOUNT,
            "NUMBER": EntityType.AMOUNT,
            "PRODUCT": EntityType.PRODUCT,
            "SERVICE": EntityType.PRODUCT,
            "LOCATION": EntityType.LOCATION,
            "PLACE": EntityType.LOCATION,
            "ADDRESS": EntityType.LOCATION,
        }
        return type_map.get(ent_type.upper(), EntityType.OTHER)
    
    def _normalize_relation_type(self, rel_type: str) -> str:
        """Normalize relationship type to standard values."""
        type_map = {
            "AUTHORED": RelationType.AUTHORED,
            "WROTE": RelationType.AUTHORED,
            "CREATED": RelationType.AUTHORED,
            "WORKS_FOR": RelationType.WORKS_FOR,
            "EMPLOYED_BY": RelationType.WORKS_FOR,
            "REFERENCES": RelationType.REFERENCES,
            "MENTIONS": RelationType.MENTIONS,
            "DATED": RelationType.DATED,
            "ON_DATE": RelationType.DATED,
            "OWNS": RelationType.OWNS,
            "HAS": RelationType.OWNS,
            "SIGNED": RelationType.SIGNED,
            "PAID": RelationType.PAID,
            "RECEIVED": RelationType.RECEIVED,
        }
        return type_map.get(rel_type.upper(), RelationType.RELATED_TO)
    
    async def extract_and_store(
        self,
        chunk_text: str,
        chunk_id: str,
        client_id: str,
        source_name: str = "unknown",
    ) -> ExtractionResult:
        """
        Extract entities and store them in the knowledge graph.
        
        Args:
            chunk_text: The document chunk text
            chunk_id: Unique identifier for the chunk
            client_id: Client ID for the knowledge graph
            source_name: Name of the source document
            
        Returns:
            ExtractionResult
        """
        # Extract
        result = await self.extract(chunk_text, chunk_id, source_name)
        
        if not result.entities:
            return result
        
        # Store in knowledge graph
        kg = get_knowledge_graph(client_id)
        
        # Add entities (check for duplicates by name)
        entity_id_map: Dict[str, str] = {}  # old_id -> final_id
        
        for entity in result.entities:
            # Check if entity already exists
            existing = kg.find_entity_by_name(entity.name, entity.type)
            if existing:
                entity_id_map[entity.id] = existing.id
            else:
                kg.add_entity(entity)
                entity_id_map[entity.id] = entity.id
        
        # Add relationships with remapped IDs
        for rel in result.relationships:
            rel.source_id = entity_id_map.get(rel.source_id, rel.source_id)
            rel.target_id = entity_id_map.get(rel.target_id, rel.target_id)
            kg.add_relationship(rel)
        
        # Add mentions with remapped IDs
        for mention in result.mentions:
            mention.entity_id = entity_id_map.get(mention.entity_id, mention.entity_id)
            kg.add_mention(mention)
        
        logger.info(
            f"Stored {len(result.entities)} entities, {len(result.relationships)} relationships "
            f"for client {client_id}"
        )
        
        return result


class RuleBasedExtractor:
    """
    Fast rule-based entity extraction (no LLM).
    
    Uses regex patterns to extract common entity types.
    Use as a complement to LLM extraction for well-structured data.
    """
    
    PATTERNS = {
        EntityType.AMOUNT: [
            r'\$[\d,]+(?:\.\d{2})?',  # $1,234.56
            r'USD\s*[\d,]+(?:\.\d{2})?',  # USD 1234.56
            r'\d+(?:\.\d+)?\s*%',  # 12.5%
        ],
        EntityType.DATE: [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
        ],
        EntityType.ORGANIZATION: [
            r'(?:[A-Z][a-z]+\s)+(?:Inc|LLC|Corp|Ltd|Company|Co|Corporation)\.?',
        ],
    }
    
    def __init__(self):
        self._compiled = {
            ent_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for ent_type, patterns in self.PATTERNS.items()
        }
    
    def extract(self, text: str, chunk_id: str) -> ExtractionResult:
        """Extract entities using regex patterns."""
        entities: List[Entity] = []
        mentions: List[Mention] = []
        
        for ent_type, patterns in self._compiled.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = Entity.create(
                        name=match.group(),
                        type=ent_type,
                        pattern_matched=pattern.pattern,
                    )
                    entities.append(entity)
                    mentions.append(Mention(
                        entity_id=entity.id,
                        doc_chunk_id=chunk_id,
                        span_start=match.start(),
                        span_end=match.end(),
                        context=text[max(0, match.start()-50):match.end()+50],
                    ))
        
        return ExtractionResult(
            entities=entities,
            relationships=[],  # Rule-based doesn't extract relationships
            mentions=mentions,
            raw_extraction={"method": "rule_based"},
        )


# Factory functions
_extractor_instance: Optional[EntityExtractor] = None


def get_entity_extractor(lm_client: Optional[LMStudioClient] = None) -> EntityExtractor:
    """Get or create the entity extractor singleton."""
    global _extractor_instance
    
    if _extractor_instance is None:
        if lm_client is None:
            lm_client = get_lmstudio_client()
        
        _extractor_instance = EntityExtractor(lm_client=lm_client)
    
    return _extractor_instance


def get_rule_based_extractor() -> RuleBasedExtractor:
    """Get a rule-based extractor for fast extraction."""
    return RuleBasedExtractor()
