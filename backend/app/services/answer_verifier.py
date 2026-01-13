"""
Answer Verification - Cross-check LLM responses against retrieved context.

This module provides a verification layer that:
1. Checks if claims in the answer are supported by the context
2. Identifies hallucinations (unsupported claims)
3. Computes a grounding confidence score
4. Adds disclaimers for unverifiable content

Critical for ensuring RAG output quality with small models.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.clients.lmstudio import LMStudioClient
from app.models.schemas import RetrievalHit

logger = logging.getLogger(__name__)


@dataclass
class Claim:
    """A factual claim extracted from an answer."""
    text: str
    supported: str  # "YES", "NO", "PARTIAL"
    evidence: Optional[str] = None
    source_id: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of answer verification."""
    is_grounded: bool           # Overall: is answer grounded in context?
    confidence: float           # 0-1 confidence score
    claims: List[Claim]         # Individual claims and their support status
    unsupported_claims: List[str]  # Claims not found in context
    partial_claims: List[str]   # Claims only partially supported
    verification_notes: str     # Human-readable summary
    disclaimer: Optional[str] = None  # Disclaimer to append if needed


VERIFICATION_PROMPT = """You are a fact-checker. Analyze if each claim in the ANSWER is supported by the CONTEXT.

CONTEXT:
{context}

ANSWER TO VERIFY:
{answer}

For each factual claim in the answer:
1. Identify the claim
2. Check if it's supported by the context (YES/NO/PARTIAL)
3. Quote the supporting evidence if found

Output ONLY valid JSON (no markdown, no explanation):
{{
  "claims": [
    {{"claim": "statement from answer", "supported": "YES|NO|PARTIAL", "evidence": "quote from context or null"}}
  ],
  "overall_grounded": true or false,
  "confidence": 0.0 to 1.0
}}

Rules:
- YES = claim is directly supported by context
- PARTIAL = claim is related but not exact match  
- NO = claim cannot be verified from context
- overall_grounded = true if all important claims are YES or PARTIAL
- confidence = (YES + 0.5*PARTIAL) / total_claims"""


class AnswerVerifier:
    """
    Verify LLM answers against retrieved context.
    
    Uses an LLM to analyze whether claims in the answer
    are supported by the provided context documents.
    """
    
    def __init__(
        self,
        lm_client: LMStudioClient,
        min_confidence: float = 0.7,
    ):
        """
        Initialize verifier.
        
        Args:
            lm_client: LLM client for verification
            min_confidence: Minimum confidence to pass verification
        """
        self.lm_client = lm_client
        self.min_confidence = min_confidence
    
    async def verify(
        self,
        answer: str,
        context: List[RetrievalHit],
    ) -> VerificationResult:
        """
        Verify an answer against the retrieved context.
        
        Args:
            answer: The LLM-generated answer to verify
            context: Retrieved documents used to generate answer
            
        Returns:
            VerificationResult with detailed analysis
        """
        if not context:
            return VerificationResult(
                is_grounded=False,
                confidence=0.0,
                claims=[],
                unsupported_claims=[],
                partial_claims=[],
                verification_notes="No context provided for verification",
                disclaimer="⚠️ This response could not be verified against source documents."
            )
        
        # Format context
        context_text = self._format_context(context)
        
        # Build verification prompt
        prompt = VERIFICATION_PROMPT.format(
            context=context_text,
            answer=answer
        )
        
        try:
            # Call LLM for verification
            response = await self.lm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            
            # Parse response
            result = self._parse_verification_response(response)
            
            # Add disclaimer if needed
            if not result.is_grounded or result.confidence < self.min_confidence:
                result.disclaimer = self._generate_disclaimer(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                is_grounded=True,  # Fail open - don't block on errors
                confidence=0.5,
                claims=[],
                unsupported_claims=[],
                partial_claims=[],
                verification_notes=f"Verification error: {str(e)}"
            )
    
    def _format_context(self, context: List[RetrievalHit]) -> str:
        """Format context documents for the verification prompt."""
        parts = []
        for hit in context:
            source = hit.metadata.get("source", hit.id)
            parts.append(f"[{source}]:\n{hit.content}")
        return "\n\n".join(parts)
    
    def _parse_verification_response(self, response: str) -> VerificationResult:
        """Parse the JSON response from the verification LLM."""
        # Try to extract JSON from response
        json_str = self._extract_json(response)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse verification JSON: {e}")
            return VerificationResult(
                is_grounded=True,
                confidence=0.5,
                claims=[],
                unsupported_claims=[],
                partial_claims=[],
                verification_notes="Could not parse verification response"
            )
        
        # Parse claims
        claims = []
        unsupported = []
        partial = []
        
        for claim_data in data.get("claims", []):
            claim = Claim(
                text=claim_data.get("claim", ""),
                supported=claim_data.get("supported", "NO").upper(),
                evidence=claim_data.get("evidence"),
            )
            claims.append(claim)
            
            if claim.supported == "NO":
                unsupported.append(claim.text)
            elif claim.supported == "PARTIAL":
                partial.append(claim.text)
        
        # Calculate confidence if not provided
        confidence = data.get("confidence", 0.5)
        if claims and confidence == 0.5:
            yes_count = sum(1 for c in claims if c.supported == "YES")
            partial_count = sum(1 for c in claims if c.supported == "PARTIAL")
            confidence = (yes_count + 0.5 * partial_count) / len(claims)
        
        is_grounded = data.get("overall_grounded", confidence >= self.min_confidence)
        
        # Build notes
        notes = self._build_verification_notes(claims, confidence)
        
        return VerificationResult(
            is_grounded=is_grounded,
            confidence=confidence,
            claims=claims,
            unsupported_claims=unsupported,
            partial_claims=partial,
            verification_notes=notes
        )
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain other content."""
        # Try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json_match.group()
        return text
    
    def _build_verification_notes(self, claims: List[Claim], confidence: float) -> str:
        """Build human-readable verification notes."""
        if not claims:
            return "No verifiable claims found in response"
        
        yes_count = sum(1 for c in claims if c.supported == "YES")
        partial_count = sum(1 for c in claims if c.supported == "PARTIAL")
        no_count = sum(1 for c in claims if c.supported == "NO")
        
        parts = [
            f"Verified {len(claims)} claims:",
            f"- Fully supported: {yes_count}",
            f"- Partially supported: {partial_count}",
            f"- Not supported: {no_count}",
            f"- Confidence: {confidence:.1%}"
        ]
        
        return "\n".join(parts)
    
    def _generate_disclaimer(self, result: VerificationResult) -> str:
        """Generate an appropriate disclaimer based on verification result."""
        if result.unsupported_claims:
            return (
                "⚠️ Note: Some statements in this response could not be verified "
                "against the source documents. Please verify critical information."
            )
        elif result.partial_claims:
            return (
                "ℹ️ Note: Some information may be inferred or approximated from "
                "the available documents."
            )
        elif result.confidence < self.min_confidence:
            return (
                "⚠️ Note: This response has lower confidence. "
                "Please verify important details."
            )
        return None
    
    async def quick_check(
        self,
        answer: str,
        context: List[RetrievalHit],
    ) -> Tuple[bool, float]:
        """
        Quick verification check without full analysis.
        
        Returns:
            Tuple of (is_grounded, confidence)
        """
        result = await self.verify(answer, context)
        return result.is_grounded, result.confidence


class RuleBasedVerifier:
    """
    Fast rule-based verification (no LLM call).
    
    Checks for basic grounding signals without semantic analysis.
    Use as a quick pre-filter before full LLM verification.
    """
    
    # Keywords suggesting claims that need verification
    FACTUAL_KEYWORDS = [
        r'\$[\d,]+',  # Dollar amounts
        r'\d+%',       # Percentages
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # Dates
        r'\d{4}-\d{2}-\d{2}',  # ISO dates
        r'total|sum|amount|value',
        r'exactly|precisely|specifically',
    ]
    
    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_KEYWORDS]
    
    def quick_check(
        self,
        answer: str,
        context: List[RetrievalHit],
    ) -> Tuple[bool, float, List[str]]:
        """
        Quick rule-based verification.
        
        Returns:
            Tuple of (likely_grounded, confidence, potential_issues)
        """
        issues = []
        
        # Extract all factual values from answer
        answer_facts = self._extract_facts(answer)
        
        # Combine context text
        context_text = " ".join(hit.content for hit in context)
        context_facts = self._extract_facts(context_text)
        
        # Check if answer facts appear in context
        unverified = []
        for fact in answer_facts:
            if fact not in context_text and fact not in str(context_facts):
                unverified.append(fact)
        
        if unverified:
            issues.append(f"Unverified facts: {unverified[:3]}")
        
        # Calculate confidence
        if answer_facts:
            verified = len(answer_facts) - len(unverified)
            confidence = verified / len(answer_facts)
        else:
            confidence = 1.0  # No factual claims to verify
        
        return confidence >= 0.7, confidence, issues
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual values from text."""
        facts = []
        for pattern in self._patterns:
            facts.extend(pattern.findall(text))
        return facts


# Factory functions
_verifier_instance: Optional[AnswerVerifier] = None


def get_answer_verifier(lm_client: Optional[LMStudioClient] = None) -> AnswerVerifier:
    """Get or create the answer verifier singleton."""
    global _verifier_instance
    
    if _verifier_instance is None:
        if lm_client is None:
            from app.clients.lmstudio import get_lmstudio_client
            lm_client = get_lmstudio_client()
        
        from app.config import settings
        _verifier_instance = AnswerVerifier(
            lm_client=lm_client,
            min_confidence=settings.rag.min_confidence_threshold,
        )
    
    return _verifier_instance


def get_rule_based_verifier() -> RuleBasedVerifier:
    """Get a rule-based verifier for quick checks."""
    return RuleBasedVerifier()
