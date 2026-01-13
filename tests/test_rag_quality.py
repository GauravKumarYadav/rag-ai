"""
Tests for RAG Quality Assurance Components.

Tests cover:
- BM25 indexing and hybrid search
- Citation extraction and validation
- Answer verification
- Knowledge graph operations
- Evaluation metrics
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import RetrievalHit


# ============================================================================
# BM25 Index Tests
# ============================================================================

class TestBM25Index:
    """Tests for BM25 keyword search."""
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        from app.rag.bm25_index import BM25Index
        
        index = BM25Index()
        tokens = index._tokenize("Hello World! This is a test.")
        
        # Should remove stopwords and short tokens
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "is" not in tokens  # Stopword
        assert "a" not in tokens   # Stopword
    
    def test_add_and_search(self):
        """Test adding documents and searching."""
        from app.rag.bm25_index import BM25Index
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index = BM25Index(persist_path=tmpdir)
            
            # Add documents
            index.add_documents(
                contents=[
                    "The quick brown fox jumps over the lazy dog",
                    "A contract worth $50,000 was signed on January 15, 2024",
                    "Machine learning models require training data",
                ],
                ids=["doc1", "doc2", "doc3"],
                metadatas=[
                    {"source": "test1.txt"},
                    {"source": "contract.pdf"},
                    {"source": "ml_guide.txt"},
                ]
            )
            
            # Search for contract
            results = index.search("contract signed", top_k=2)
            
            assert len(results) > 0
            assert results[0].id == "doc2"  # Contract doc should be first
    
    def test_delete_documents(self):
        """Test deleting documents."""
        from app.rag.bm25_index import BM25Index
        
        index = BM25Index()
        
        index.add_documents(
            contents=["Document one", "Document two"],
            ids=["d1", "d2"],
        )
        
        assert len(index.documents) == 2
        
        deleted = index.delete(ids=["d1"])
        
        assert deleted == 1
        assert len(index.documents) == 1
        assert "d2" in index.documents
    
    def test_persistence(self):
        """Test saving and loading index."""
        from app.rag.bm25_index import BM25Index
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate index
            index1 = BM25Index(persist_path=tmpdir, client_id="test_client")
            index1.add_documents(
                contents=["Persistent document content"],
                ids=["pdoc1"],
            )
            index1.persist()
            
            # Load in new instance
            index2 = BM25Index(persist_path=tmpdir, client_id="test_client")
            
            assert len(index2.documents) == 1
            assert "pdoc1" in index2.documents


class TestHybridSearch:
    """Tests for hybrid BM25 + Vector search."""
    
    def test_rrf_fusion(self):
        """Test Reciprocal Rank Fusion."""
        from app.rag.hybrid_search import HybridSearch
        
        # Mock vector store and BM25
        mock_vector = MagicMock()
        mock_bm25 = MagicMock()
        
        hybrid = HybridSearch(
            vector_store=mock_vector,
            bm25_index=mock_bm25,
            reranker=None,
            bm25_weight=0.4,
            vector_weight=0.6,
        )
        
        # Create hits
        vector_hits = [
            RetrievalHit(id="doc1", content="Content 1", score=0.1, metadata={}),
            RetrievalHit(id="doc2", content="Content 2", score=0.2, metadata={}),
        ]
        bm25_hits = [
            RetrievalHit(id="doc2", content="Content 2", score=0.15, metadata={}),
            RetrievalHit(id="doc3", content="Content 3", score=0.3, metadata={}),
        ]
        
        fused = hybrid._rrf_fusion(vector_hits, bm25_hits)
        
        # doc2 should be ranked highest (appears in both)
        assert len(fused) == 3
        assert fused[0].id == "doc2"


# ============================================================================
# Citation Extractor Tests
# ============================================================================

class TestCitationExtractor:
    """Tests for citation extraction."""
    
    def test_extract_citations(self):
        """Test extracting citations from response."""
        from app.services.citation_extractor import CitationExtractor
        
        extractor = CitationExtractor()
        
        response = (
            "The contract value is $50,000 [Source: contract.pdf#chunk-2] "
            "and expires on December 31, 2024 [Source: contract.pdf#chunk-5]."
        )
        
        citations = extractor.extract_citations(response)
        
        assert len(citations) == 2
        assert citations[0].source_id == "contract.pdf#chunk-2"
        assert citations[1].source_id == "contract.pdf#chunk-5"
    
    def test_find_uncited_claims(self):
        """Test finding uncited factual claims."""
        from app.services.citation_extractor import CitationExtractor
        
        extractor = CitationExtractor()
        
        response = (
            "The contract value is $50,000 [Source: contract.pdf]. "
            "The deadline is January 15, 2024. "  # Uncited
            "The company earned $1M last year."    # Uncited
        )
        
        uncited = extractor.find_uncited_claims(response)
        
        # Should find uncited factual claims
        assert len(uncited) > 0
    
    def test_analyze_with_retrieved_docs(self):
        """Test full citation analysis with context validation."""
        from app.services.citation_extractor import CitationExtractor
        
        extractor = CitationExtractor(min_coverage=0.5)
        
        response = "Value is $100 [Source: doc1]"
        retrieved = [
            RetrievalHit(id="doc1", content="Value info", score=0.1, metadata={"source": "doc1"}),
        ]
        
        analysis = extractor.analyze(response, retrieved)
        
        assert analysis.citation_count == 1
        assert len(analysis.valid_citations) == 1
        assert len(analysis.invalid_citations) == 0


# ============================================================================
# Answer Verifier Tests
# ============================================================================

class TestAnswerVerifier:
    """Tests for answer verification."""
    
    @pytest.mark.asyncio
    async def test_verify_grounded_answer(self):
        """Test verification of well-grounded answer."""
        from app.services.answer_verifier import AnswerVerifier
        
        # Mock LLM client
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=json.dumps({
            "claims": [
                {"claim": "The value is $50,000", "supported": "YES", "evidence": "contract shows $50,000"}
            ],
            "overall_grounded": True,
            "confidence": 0.95
        }))
        
        verifier = AnswerVerifier(lm_client=mock_client)
        
        answer = "The contract value is $50,000."
        context = [
            RetrievalHit(
                id="doc1", 
                content="Contract value: $50,000", 
                score=0.1, 
                metadata={"source": "contract.pdf"}
            )
        ]
        
        result = await verifier.verify(answer, context)
        
        assert result.is_grounded
        assert result.confidence >= 0.9
        assert len(result.unsupported_claims) == 0
    
    @pytest.mark.asyncio
    async def test_verify_empty_context(self):
        """Test verification with no context."""
        from app.services.answer_verifier import AnswerVerifier
        
        mock_client = AsyncMock()
        verifier = AnswerVerifier(lm_client=mock_client)
        
        result = await verifier.verify("Some answer", [])
        
        assert not result.is_grounded
        assert result.confidence == 0.0
        assert result.disclaimer is not None
    
    def test_rule_based_verifier(self):
        """Test rule-based verification."""
        from app.services.answer_verifier import RuleBasedVerifier
        
        verifier = RuleBasedVerifier()
        
        answer = "The amount is $50,000 paid on 01/15/2024"
        context = [
            RetrievalHit(
                id="doc1",
                content="Payment of $50,000 on 01/15/2024",
                score=0.1,
                metadata={}
            )
        ]
        
        grounded, confidence, issues = verifier.quick_check(answer, context)
        
        assert grounded  # Facts appear in context
        assert confidence > 0.5


# ============================================================================
# Knowledge Graph Tests
# ============================================================================

class TestKnowledgeGraphStore:
    """Tests for knowledge graph storage."""
    
    def test_add_and_get_entity(self):
        """Test adding and retrieving entities."""
        from app.knowledge.graph_store import KnowledgeGraphStore, Entity
        
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraphStore(client_id="test", persist_path=tmpdir)
            
            entity = Entity.create(name="John Doe", type="PERSON", role="Manager")
            kg.add_entity(entity)
            
            retrieved = kg.get_entity(entity.id)
            
            assert retrieved is not None
            assert retrieved.name == "John Doe"
            assert retrieved.type == "PERSON"
            
            kg.close()
    
    def test_add_and_query_relationships(self):
        """Test relationship operations."""
        from app.knowledge.graph_store import KnowledgeGraphStore, Entity, Relationship
        
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraphStore(client_id="test", persist_path=tmpdir)
            
            # Create entities
            person = Entity.create(name="Jane Smith", type="PERSON")
            company = Entity.create(name="Acme Corp", type="ORG")
            
            kg.add_entity(person)
            kg.add_entity(company)
            
            # Create relationship
            rel = Relationship.create(
                source_id=person.id,
                target_id=company.id,
                relation_type="WORKS_FOR"
            )
            kg.add_relationship(rel)
            
            # Query relationships
            relationships = kg.get_relationships(person.id, direction="outgoing")
            
            assert len(relationships) == 1
            assert relationships[0][0].relation_type == "WORKS_FOR"
            
            kg.close()
    
    def test_get_related_entities(self):
        """Test multi-hop relationship traversal."""
        from app.knowledge.graph_store import KnowledgeGraphStore, Entity, Relationship
        
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraphStore(client_id="test", persist_path=tmpdir)
            
            # Create chain: Person -> Company -> Contract
            person = Entity.create(name="Alice", type="PERSON")
            company = Entity.create(name="TechCorp", type="ORG")
            contract = Entity.create(name="Service Agreement", type="DOCUMENT")
            
            for e in [person, company, contract]:
                kg.add_entity(e)
            
            # Person works for company
            kg.add_relationship(Relationship.create(
                source_id=person.id, target_id=company.id, relation_type="WORKS_FOR"
            ))
            # Company has contract
            kg.add_relationship(Relationship.create(
                source_id=company.id, target_id=contract.id, relation_type="OWNS"
            ))
            
            # Query 2-hop relationships from person
            related = kg.get_related_entities(person.id, depth=2)
            
            # Should find both company and contract
            related_names = [e.name for e in related]
            assert "TechCorp" in related_names
            assert "Service Agreement" in related_names
            
            kg.close()


class TestEntityExtractor:
    """Tests for entity extraction."""
    
    @pytest.mark.asyncio
    async def test_extract_entities(self):
        """Test entity extraction from document chunk."""
        from app.knowledge.entity_extractor import EntityExtractor
        
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=json.dumps({
            "entities": [
                {"name": "John Smith", "type": "PERSON", "attributes": {"role": "CEO"}},
                {"name": "Acme Inc", "type": "ORG", "attributes": {}},
                {"name": "$100,000", "type": "AMOUNT", "attributes": {}},
            ],
            "relationships": [
                {"source": "John Smith", "target": "Acme Inc", "type": "WORKS_FOR"}
            ]
        }))
        
        extractor = EntityExtractor(lm_client=mock_client)
        
        result = await extractor.extract(
            chunk_text="John Smith, CEO of Acme Inc, signed a $100,000 contract.",
            chunk_id="chunk-001",
            source_name="contract.pdf"
        )
        
        assert len(result.entities) == 3
        assert len(result.relationships) == 1
        assert len(result.mentions) == 3
    
    def test_rule_based_extraction(self):
        """Test rule-based entity extraction."""
        from app.knowledge.entity_extractor import RuleBasedExtractor
        
        extractor = RuleBasedExtractor()
        
        result = extractor.extract(
            text="Payment of $50,000 on 01/15/2024 to Acme Corp.",
            chunk_id="chunk-001"
        )
        
        # Should find amount and date
        entity_types = [e.type for e in result.entities]
        assert "AMOUNT" in entity_types
        assert "DATE" in entity_types


# ============================================================================
# Evaluation Metrics Tests
# ============================================================================

class TestRAGEvaluator:
    """Tests for RAG evaluation metrics."""
    
    @pytest.mark.asyncio
    async def test_evaluate_single(self):
        """Test single test case evaluation."""
        from app.evaluation.quality_metrics import RAGEvaluator, TestCase
        
        mock_client = AsyncMock()
        # Mock verification response
        mock_client.chat = AsyncMock(side_effect=[
            json.dumps({
                "claims": [{"claim": "test", "supported": "YES", "evidence": "..."}],
                "overall_grounded": True,
                "confidence": 0.9
            }),
            "0.85"  # Relevance score
        ])
        
        mock_verifier = AsyncMock()
        mock_verifier.verify = AsyncMock(return_value=MagicMock(
            is_grounded=True,
            confidence=0.9,
            unsupported_claims=[],
            disclaimer=None
        ))
        
        mock_citation = MagicMock()
        mock_citation.analyze = MagicMock(return_value=MagicMock(
            citations=[],
            valid_citations=[],
            invalid_citations=[],
            coverage_ratio=1.0
        ))
        
        evaluator = RAGEvaluator(
            lm_client=mock_client,
            verifier=mock_verifier,
            citation_extractor=mock_citation
        )
        
        test_case = TestCase(
            id="test1",
            query="What is the contract value?",
            expected_sources=["doc1"],
            expected_answer_contains=["$50,000"],
        )
        
        result = await evaluator.evaluate_single(
            test_case=test_case,
            response="The contract value is $50,000 [Source: doc1]",
            retrieved=[
                RetrievalHit(id="doc1", content="Contract: $50,000", score=0.1, metadata={})
            ]
        )
        
        assert result.hit
        assert result.reciprocal_rank == 1.0
        assert result.faithfulness > 0.5
    
    def test_test_dataset_builder(self):
        """Test building test datasets."""
        from app.evaluation.quality_metrics import TestDatasetBuilder, TestCase
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {
                    "id": "test1",
                    "query": "What is X?",
                    "expected_sources": ["doc1"],
                    "expected_answer_contains": ["fact1"],
                    "client_id": "client1",
                    "category": "factual",
                    "difficulty": "easy"
                }
            ], f)
            f.flush()
            
            cases = TestDatasetBuilder.from_json(f.name)
            
            assert len(cases) == 1
            assert cases[0].query == "What is X?"
            assert cases[0].category == "factual"


# ============================================================================
# Integration Tests
# ============================================================================

class TestChatServiceIntegration:
    """Integration tests for chat service with new components."""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_integration(self):
        """Test that hybrid search is used when enabled."""
        # This would require more extensive mocking
        # For now, just verify the import paths work
        from app.services.chat_service import ChatService
        from app.rag.hybrid_search import get_hybrid_search
        
        assert ChatService is not None
        assert get_hybrid_search is not None
    
    @pytest.mark.asyncio
    async def test_citation_verification_integration(self):
        """Test citation extraction and verification integration."""
        from app.services.citation_extractor import get_citation_extractor
        from app.services.answer_verifier import get_rule_based_verifier
        
        extractor = get_citation_extractor()
        verifier = get_rule_based_verifier()
        
        response = "The value is $50,000 [Source: contract.pdf]"
        context = [
            RetrievalHit(id="doc1", content="$50,000 contract", score=0.1, metadata={"source": "contract.pdf"})
        ]
        
        # Citation extraction
        analysis = extractor.analyze(response, context)
        assert analysis.citation_count == 1
        
        # Rule-based verification
        grounded, confidence, _ = verifier.quick_check(response, context)
        assert confidence > 0.5
