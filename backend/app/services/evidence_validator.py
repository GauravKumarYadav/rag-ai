"""
Evidence Validation for Small Model RAG Optimization.

This module provides guardrails to prevent hallucination:
1. Evidence thresholding - measure retrieval confidence
2. Contradiction detection - flag conflicting information
3. Confidence scoring - help decide when to add disclaimers

Small models confidently answer even with weak evidence.
These guardrails help identify when evidence is insufficient.
"""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

from app.config import settings
from app.models.schemas import RetrievalHit
from app.services.context_compressor import CompressedFact

logger = logging.getLogger(__name__)


@dataclass
class EvidenceAssessment:
    """Assessment of evidence quality for a query."""
    confidence: float  # 0.0 to 1.0
    has_sufficient_evidence: bool
    top_score: float
    score_gap: float  # Gap between top1 and top5
    num_relevant_hits: int
    recommendation: str  # "answer", "partial", "clarify", "decline"
    disclaimer: Optional[str]  # Suggested disclaimer if needed


@dataclass  
class Contradiction:
    """A detected contradiction between facts."""
    fact1: CompressedFact
    fact2: CompressedFact
    contradiction_type: str  # "numeric", "temporal", "logical", "semantic"
    description: str
    confidence: float


class EvidenceThresholder:
    """
    Computes confidence scores for retrieval results.
    
    Uses multiple signals:
    - Top similarity score
    - Gap between top scores (larger gap = clearer winner)
    - Number of relevant hits
    - Score distribution
    """
    
    def __init__(
        self,
        min_confidence: float = 0.3,
        min_top_score: float = 0.5,  # For ChromaDB distance (lower is better)
        good_gap_threshold: float = 0.1,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_top_score = min_top_score
        self.good_gap_threshold = good_gap_threshold
    
    def assess(self, hits: List[RetrievalHit]) -> EvidenceAssessment:
        """
        Assess evidence quality from retrieval hits.
        
        Note: ChromaDB uses L2 distance, so lower scores = more similar.
        For cosine similarity systems, higher = better.
        """
        if not hits:
            return EvidenceAssessment(
                confidence=0.0,
                has_sufficient_evidence=False,
                top_score=0.0,
                score_gap=0.0,
                num_relevant_hits=0,
                recommendation="clarify",
                disclaimer="I don't have any relevant information to answer this question.",
            )
        
        # Get scores (assuming lower = better for ChromaDB distance)
        scores = [hit.score for hit in hits]
        top_score = scores[0]
        
        # Calculate gap between top and 5th result
        if len(scores) >= 5:
            score_gap = scores[4] - scores[0]  # Higher gap = clearer signal
        else:
            score_gap = scores[-1] - scores[0] if len(scores) > 1 else 0.0
        
        # Count "relevant" hits (those within 2x of best score)
        relevance_threshold = top_score * 2  # Adjust based on your embedding
        num_relevant = sum(1 for s in scores if s <= relevance_threshold)
        
        # Compute confidence score
        confidence = self._compute_confidence(
            top_score=top_score,
            score_gap=score_gap,
            num_relevant=num_relevant,
            total_hits=len(hits),
        )
        
        # Determine recommendation
        has_sufficient = confidence >= self.min_confidence
        recommendation, disclaimer = self._get_recommendation(
            confidence=confidence,
            has_sufficient=has_sufficient,
            num_relevant=num_relevant,
        )
        
        return EvidenceAssessment(
            confidence=confidence,
            has_sufficient_evidence=has_sufficient,
            top_score=top_score,
            score_gap=score_gap,
            num_relevant_hits=num_relevant,
            recommendation=recommendation,
            disclaimer=disclaimer,
        )
    
    def _compute_confidence(
        self,
        top_score: float,
        score_gap: float,
        num_relevant: int,
        total_hits: int,
    ) -> float:
        """
        Compute overall confidence from multiple signals.
        
        Returns value between 0.0 and 1.0.
        """
        # Score component: lower distance = higher confidence
        # Assuming typical ChromaDB distances are 0.0-2.0
        score_confidence = max(0.0, 1.0 - (top_score / 2.0))
        
        # Gap component: larger gap between top and rest = clearer signal
        gap_confidence = min(1.0, score_gap / 0.5) if score_gap > 0 else 0.5
        
        # Coverage component: more relevant hits = more confidence
        coverage_confidence = min(1.0, num_relevant / 3)
        
        # Weighted combination
        confidence = (
            score_confidence * 0.5 +
            gap_confidence * 0.3 +
            coverage_confidence * 0.2
        )
        
        return round(confidence, 3)
    
    def _get_recommendation(
        self,
        confidence: float,
        has_sufficient: bool,
        num_relevant: int,
    ) -> Tuple[str, Optional[str]]:
        """Get recommendation and disclaimer based on confidence."""
        
        if confidence >= 0.7:
            return "answer", None
        
        elif confidence >= 0.5:
            return "answer", "Based on the available documents, "
        
        elif confidence >= 0.3:
            if num_relevant >= 2:
                return "partial", (
                    "I found some relevant information, but my confidence is limited. "
                    "Please verify this with authoritative sources."
                )
            else:
                return "clarify", (
                    "I found limited information on this topic. "
                    "Could you provide more context or rephrase your question?"
                )
        
        else:
            return "decline", (
                "I don't have sufficient information in the provided documents "
                "to answer this question confidently."
            )


class ContradictionDetector:
    """
    Detects contradictions between compressed facts.
    
    Types of contradictions:
    - Numeric: conflicting numbers/percentages
    - Temporal: conflicting dates/times
    - Logical: direct logical opposition
    - Semantic: semantically conflicting statements
    """
    
    # Patterns for numeric values
    NUMBER_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?)\s*(%|percent|days?|hours?|minutes?|years?|months?|weeks?|\$|dollars?|euros?|pounds?)?\b', re.IGNORECASE)
    
    # Patterns for dates
    DATE_PATTERN = re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|january|february|march|april|may|june|july|august|september|october|november|december)\b', re.IGNORECASE)
    
    # Negation patterns
    NEGATION_PATTERNS = [
        (r'\bnot\s+', r'\b(?!not\s+)'),
        (r'\bno\s+', r'\b(?!no\s+)'),
        (r'\bnever\s+', r'\balways\s+'),
        (r'\brequired\b', r'\boptional\b'),
        (r'\bmust\b', r'\bmay\b'),
        (r'\ballowed\b', r'\bprohibited\b'),
    ]
    
    def __init__(self, lm_client: Optional[object] = None) -> None:
        self.lm_client = lm_client
    
    async def detect(self, facts: List[CompressedFact]) -> List[Contradiction]:
        """
        Detect contradictions between facts.
        
        Returns list of detected contradictions.
        """
        if len(facts) < 2:
            return []
        
        contradictions = []
        
        # Compare all pairs
        for i, fact1 in enumerate(facts):
            for fact2 in facts[i+1:]:
                contradiction = self._check_pair(fact1, fact2)
                if contradiction:
                    contradictions.append(contradiction)
        
        # Optionally use LLM for deeper semantic checking
        if self.lm_client and len(facts) > 2:
            llm_contradictions = await self._llm_check(facts)
            # Merge, avoiding duplicates
            for c in llm_contradictions:
                if not any(self._is_same_contradiction(c, existing) 
                          for existing in contradictions):
                    contradictions.append(c)
        
        return contradictions
    
    def _check_pair(
        self,
        fact1: CompressedFact,
        fact2: CompressedFact,
    ) -> Optional[Contradiction]:
        """Check if two facts contradict each other."""
        text1 = fact1.text.lower()
        text2 = fact2.text.lower()
        
        # Check for numeric contradictions
        nums1 = self._extract_numbers(fact1.text)
        nums2 = self._extract_numbers(fact2.text)
        
        if nums1 and nums2:
            # Check if they're about the same thing but with different numbers
            for n1 in nums1:
                for n2 in nums2:
                    if n1['unit'] == n2['unit'] and n1['value'] != n2['value']:
                        # Check context similarity
                        if self._similar_context(text1, text2):
                            return Contradiction(
                                fact1=fact1,
                                fact2=fact2,
                                contradiction_type="numeric",
                                description=f"Conflicting values: {n1['value']}{n1['unit'] or ''} vs {n2['value']}{n2['unit'] or ''}",
                                confidence=0.7,
                            )
        
        # Check for negation patterns
        for pos_pattern, neg_pattern in self.NEGATION_PATTERNS:
            if (re.search(pos_pattern, text1) and re.search(neg_pattern, text2)) or \
               (re.search(neg_pattern, text1) and re.search(pos_pattern, text2)):
                if self._similar_context(text1, text2):
                    return Contradiction(
                        fact1=fact1,
                        fact2=fact2,
                        contradiction_type="logical",
                        description="Potentially conflicting statements",
                        confidence=0.6,
                    )
        
        return None
    
    def _extract_numbers(self, text: str) -> List[dict]:
        """Extract numbers with their units from text."""
        matches = self.NUMBER_PATTERN.findall(text)
        results = []
        for match in matches:
            value = float(match[0])
            unit = match[1].lower() if match[1] else None
            results.append({'value': value, 'unit': unit})
        return results
    
    def _similar_context(self, text1: str, text2: str) -> bool:
        """Check if two texts have similar context (same topic)."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                       'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from'}
        words1 = words1 - common_words
        words2 = words2 - common_words
        
        if not words1 or not words2:
            return False
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0
        
        return similarity > 0.3
    
    async def _llm_check(self, facts: List[CompressedFact]) -> List[Contradiction]:
        """Use LLM to check for semantic contradictions."""
        if not self.lm_client:
            return []
        
        try:
            facts_text = "\n".join(f"{i+1}. {f.text}" for i, f in enumerate(facts))
            
            prompt = f"""Analyze these facts for contradictions. List any pairs that contradict each other.

Facts:
{facts_text}

Output format (only if contradictions found):
- Fact X contradicts Fact Y: [brief explanation]

If no contradictions, respond with: "No contradictions found."
"""
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.lm_client.chat(messages, stream=False)
            
            # Parse response for contradictions
            return self._parse_llm_response(response, facts)
            
        except Exception as e:
            logger.warning(f"LLM contradiction check failed: {e}")
            return []
    
    def _parse_llm_response(
        self,
        response: str,
        facts: List[CompressedFact],
    ) -> List[Contradiction]:
        """Parse LLM response for contradictions."""
        contradictions = []
        
        if "no contradictions" in response.lower():
            return []
        
        # Look for patterns like "Fact X contradicts Fact Y"
        pattern = r'fact\s*(\d+)\s*contradicts?\s*fact\s*(\d+)'
        matches = re.findall(pattern, response.lower())
        
        for match in matches:
            try:
                idx1 = int(match[0]) - 1
                idx2 = int(match[1]) - 1
                
                if 0 <= idx1 < len(facts) and 0 <= idx2 < len(facts):
                    contradictions.append(Contradiction(
                        fact1=facts[idx1],
                        fact2=facts[idx2],
                        contradiction_type="semantic",
                        description="LLM detected semantic contradiction",
                        confidence=0.5,
                    ))
            except (ValueError, IndexError):
                continue
        
        return contradictions
    
    def _is_same_contradiction(self, c1: Contradiction, c2: Contradiction) -> bool:
        """Check if two contradictions are about the same facts."""
        ids1 = {c1.fact1.source_id, c1.fact2.source_id}
        ids2 = {c2.fact1.source_id, c2.fact2.source_id}
        return ids1 == ids2


class EvidenceValidator:
    """
    Main evidence validator combining thresholding and contradiction detection.
    
    Usage:
        validator = EvidenceValidator()
        assessment = validator.assess_evidence(hits)
        contradictions = await validator.check_contradictions(facts)
    """
    
    def __init__(
        self,
        lm_client: Optional[object] = None,
        min_confidence: Optional[float] = None,
    ) -> None:
        min_conf = min_confidence or settings.rag.min_confidence_threshold
        self.thresholder = EvidenceThresholder(min_confidence=min_conf)
        self.contradiction_detector = ContradictionDetector(lm_client=lm_client)
    
    def assess_evidence(self, hits: List[RetrievalHit]) -> EvidenceAssessment:
        """Assess evidence quality from retrieval hits."""
        return self.thresholder.assess(hits)
    
    async def check_contradictions(
        self,
        facts: List[CompressedFact],
    ) -> List[Contradiction]:
        """Check for contradictions between facts."""
        return await self.contradiction_detector.detect(facts)
    
    def format_contradictions_warning(
        self,
        contradictions: List[Contradiction],
    ) -> Optional[str]:
        """Format contradictions as a warning message for the user."""
        if not contradictions:
            return None
        
        warnings = ["Note: The following information may be conflicting:"]
        for c in contradictions[:3]:  # Limit to top 3
            warnings.append(f"- {c.description}")
        
        return "\n".join(warnings)


@lru_cache(maxsize=1)
def get_evidence_validator() -> EvidenceValidator:
    """Get singleton evidence validator (without LLM client)."""
    return EvidenceValidator()


def get_evidence_validator_with_llm(lm_client: object) -> EvidenceValidator:
    """Get evidence validator with LLM client for semantic checking."""
    return EvidenceValidator(lm_client=lm_client)
