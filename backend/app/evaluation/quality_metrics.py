"""
Enhanced RAG Quality Metrics.

Measures:
- Hit Rate: Did we retrieve the right document?
- MRR (Mean Reciprocal Rank): How high was the correct doc ranked?
- Faithfulness: Is the answer grounded in context?
- Citation Accuracy: Can we verify the citations?
- Answer Relevance: Does the answer address the query?
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional

from app.clients.lmstudio import LMStudioClient
from app.services.answer_verifier import AnswerVerifier, get_answer_verifier
from app.services.citation_extractor import CitationExtractor, get_citation_extractor
from app.models.schemas import RetrievalHit

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A test case for RAG evaluation."""
    id: str
    query: str
    expected_sources: List[str]  # Document IDs that should be retrieved
    expected_answer_contains: List[str] = field(default_factory=list)  # Key facts
    client_id: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class TestResult:
    """Result of a single test case."""
    test_case_id: str
    query: str
    
    # Retrieval metrics
    retrieved_ids: List[str]
    hit: bool  # Did we retrieve at least one expected source?
    reciprocal_rank: float  # 1/rank of first correct source
    
    # Response metrics
    response: str
    faithfulness: float  # 0-1 grounding score
    citation_accuracy: float  # Valid citations / total citations
    answer_relevance: float  # 0-1 relevance score
    
    # Details
    unsupported_claims: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    missing_facts: List[str] = field(default_factory=list)


@dataclass
class RAGQualityMetrics:
    """Aggregated RAG quality metrics."""
    # Core metrics
    hit_rate: float           # Proportion of queries that retrieved correct docs
    mrr: float                # Mean Reciprocal Rank
    faithfulness: float       # Average grounding score
    citation_accuracy: float  # Average citation validity
    answer_relevance: float   # Average relevance score
    
    # Counts
    total_queries: int
    successful_retrievals: int
    
    # Breakdown by category
    metrics_by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Timestamp
    evaluated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "faithfulness": self.faithfulness,
            "citation_accuracy": self.citation_accuracy,
            "answer_relevance": self.answer_relevance,
            "total_queries": self.total_queries,
            "successful_retrievals": self.successful_retrievals,
            "metrics_by_category": self.metrics_by_category,
            "evaluated_at": self.evaluated_at,
        }


class RAGEvaluator:
    """
    Evaluate RAG pipeline quality.
    
    Runs test cases through the pipeline and computes metrics.
    """
    
    def __init__(
        self,
        lm_client: LMStudioClient,
        verifier: Optional[AnswerVerifier] = None,
        citation_extractor: Optional[CitationExtractor] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            lm_client: LLM client for relevance scoring
            verifier: Answer verifier for faithfulness
            citation_extractor: Citation extractor for accuracy
        """
        self.lm_client = lm_client
        self.verifier = verifier or get_answer_verifier(lm_client)
        self.citation_extractor = citation_extractor or get_citation_extractor()
    
    async def evaluate_single(
        self,
        test_case: TestCase,
        response: str,
        retrieved: List[RetrievalHit],
    ) -> TestResult:
        """
        Evaluate a single test case.
        
        Args:
            test_case: The test case
            response: Generated response
            retrieved: Retrieved documents
            
        Returns:
            TestResult with all metrics
        """
        # Compute retrieval metrics
        retrieved_ids = [hit.id for hit in retrieved]
        
        # Hit: did we retrieve any expected source?
        hit = any(
            expected in retrieved_ids or any(expected in rid for rid in retrieved_ids)
            for expected in test_case.expected_sources
        )
        
        # Reciprocal Rank: 1/rank of first correct source
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids):
            if any(expected in rid or rid in expected for expected in test_case.expected_sources):
                rr = 1.0 / (rank + 1)
                break
        
        # Faithfulness via verifier
        verification = await self.verifier.verify(response, retrieved)
        faithfulness = verification.confidence
        unsupported = verification.unsupported_claims
        
        # Citation accuracy
        citation_analysis = self.citation_extractor.analyze(response, retrieved)
        citation_acc = (
            len(citation_analysis.valid_citations) / len(citation_analysis.citations)
            if citation_analysis.citations else 1.0
        )
        invalid_cites = [c.source_id for c in citation_analysis.invalid_citations]
        
        # Answer relevance via LLM
        relevance = await self._score_relevance(test_case.query, response)
        
        # Check for missing expected facts
        missing = [
            fact for fact in test_case.expected_answer_contains
            if fact.lower() not in response.lower()
        ]
        
        return TestResult(
            test_case_id=test_case.id,
            query=test_case.query,
            retrieved_ids=retrieved_ids,
            hit=hit,
            reciprocal_rank=rr,
            response=response,
            faithfulness=faithfulness,
            citation_accuracy=citation_acc,
            answer_relevance=relevance,
            unsupported_claims=unsupported,
            invalid_citations=invalid_cites,
            missing_facts=missing,
        )
    
    async def _score_relevance(self, query: str, response: str) -> float:
        """Score how relevant the response is to the query."""
        prompt = f"""Rate how well this response answers the query.

Query: {query}

Response: {response}

Rate from 0.0 to 1.0:
- 0.0 = Completely irrelevant
- 0.5 = Partially relevant
- 1.0 = Fully relevant and complete

Output ONLY a number (e.g., 0.7):"""

        try:
            result = await self.lm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            # Extract number from response
            import re
            match = re.search(r'(\d+\.?\d*)', result.strip())
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
            return 0.5
        except Exception as e:
            logger.error(f"Relevance scoring failed: {e}")
            return 0.5
    
    async def evaluate_batch(
        self,
        test_cases: List[TestCase],
        run_query_fn,  # async fn(query, client_id) -> (response, retrieved)
    ) -> RAGQualityMetrics:
        """
        Evaluate a batch of test cases.
        
        Args:
            test_cases: List of test cases
            run_query_fn: Function to run queries through RAG pipeline
            
        Returns:
            Aggregated RAGQualityMetrics
        """
        results: List[TestResult] = []
        
        for test_case in test_cases:
            try:
                # Run query through RAG
                response, retrieved = await run_query_fn(
                    test_case.query, 
                    test_case.client_id
                )
                
                # Evaluate result
                result = await self.evaluate_single(test_case, response, retrieved)
                results.append(result)
                
                logger.debug(
                    f"Evaluated {test_case.id}: hit={result.hit}, "
                    f"faithfulness={result.faithfulness:.2f}"
                )
                
            except Exception as e:
                logger.error(f"Failed to evaluate {test_case.id}: {e}")
        
        # Aggregate metrics
        return self._aggregate_results(results, test_cases)
    
    def _aggregate_results(
        self,
        results: List[TestResult],
        test_cases: List[TestCase],
    ) -> RAGQualityMetrics:
        """Aggregate individual results into metrics."""
        if not results:
            return RAGQualityMetrics(
                hit_rate=0.0,
                mrr=0.0,
                faithfulness=0.0,
                citation_accuracy=0.0,
                answer_relevance=0.0,
                total_queries=0,
                successful_retrievals=0,
            )
        
        # Core metrics
        hit_rate = mean([1.0 if r.hit else 0.0 for r in results])
        mrr = mean([r.reciprocal_rank for r in results])
        faithfulness = mean([r.faithfulness for r in results])
        citation_accuracy = mean([r.citation_accuracy for r in results])
        answer_relevance = mean([r.answer_relevance for r in results])
        
        # Successful retrievals
        successful = sum(1 for r in results if r.hit)
        
        # Metrics by category
        test_case_map = {tc.id: tc for tc in test_cases}
        categories: Dict[str, List[TestResult]] = {}
        
        for result in results:
            tc = test_case_map.get(result.test_case_id)
            if tc:
                cat = tc.category
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(result)
        
        metrics_by_category = {}
        for cat, cat_results in categories.items():
            metrics_by_category[cat] = {
                "hit_rate": mean([1.0 if r.hit else 0.0 for r in cat_results]),
                "faithfulness": mean([r.faithfulness for r in cat_results]),
                "count": len(cat_results),
            }
        
        return RAGQualityMetrics(
            hit_rate=hit_rate,
            mrr=mrr,
            faithfulness=faithfulness,
            citation_accuracy=citation_accuracy,
            answer_relevance=answer_relevance,
            total_queries=len(results),
            successful_retrievals=successful,
            metrics_by_category=metrics_by_category,
        )


class TestDatasetBuilder:
    """
    Build test datasets for RAG evaluation.
    """
    
    @staticmethod
    def from_documents(
        documents: List[Dict[str, Any]],
        client_id: str,
        queries_per_doc: int = 2,
    ) -> List[TestCase]:
        """
        Generate test cases from documents.
        
        This is a simple heuristic approach. For production,
        use human-labeled test sets or more sophisticated generation.
        """
        test_cases: List[TestCase] = []
        
        for doc in documents:
            doc_id = doc.get("id", "unknown")
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "unknown")
            
            # Simple query generation based on content
            # In production, use LLM or human-generated queries
            lines = content.split('\n')
            for i in range(min(queries_per_doc, len(lines))):
                line = lines[i].strip()
                if len(line) > 20:
                    test_cases.append(TestCase(
                        id=f"{doc_id}_q{i}",
                        query=f"What information is in {source} about: {line[:50]}...?",
                        expected_sources=[doc_id],
                        expected_answer_contains=[line[:30]] if len(line) > 30 else [],
                        client_id=client_id,
                    ))
        
        return test_cases
    
    @staticmethod
    def from_json(json_path: str) -> List[TestCase]:
        """Load test cases from JSON file."""
        with open(json_path) as f:
            data = json.load(f)
        
        return [
            TestCase(
                id=item["id"],
                query=item["query"],
                expected_sources=item.get("expected_sources", []),
                expected_answer_contains=item.get("expected_answer_contains", []),
                client_id=item.get("client_id"),
                category=item.get("category", "general"),
                difficulty=item.get("difficulty", "medium"),
            )
            for item in data
        ]
    
    @staticmethod
    def to_json(test_cases: List[TestCase], json_path: str) -> None:
        """Save test cases to JSON file."""
        data = [
            {
                "id": tc.id,
                "query": tc.query,
                "expected_sources": tc.expected_sources,
                "expected_answer_contains": tc.expected_answer_contains,
                "client_id": tc.client_id,
                "category": tc.category,
                "difficulty": tc.difficulty,
            }
            for tc in test_cases
        ]
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)


# Factory function
def get_rag_evaluator(lm_client: Optional[LMStudioClient] = None) -> RAGEvaluator:
    """Get a RAG evaluator instance."""
    if lm_client is None:
        from app.clients.lmstudio import get_lmstudio_client
        lm_client = get_lmstudio_client()
    
    return RAGEvaluator(lm_client=lm_client)
