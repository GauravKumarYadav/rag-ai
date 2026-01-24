"""
Verification Agent.

Verifies answer quality and triggers re-retrieval if needed.
Implements the self-correction loop for agentic RAG.
"""

import json
import logging
import re
from typing import List, Optional, Tuple

from app.agents.base import BaseAgent
from app.agents.state import (
    AgentAction,
    AgentState,
    ActionType,
    VerificationResult,
)
from app.clients.lmstudio import LMStudioClient
from app.config import settings
from app.models.schemas import RetrievalHit

logger = logging.getLogger(__name__)


VERIFICATION_PROMPT = """Verify if this answer is grounded in the provided context.

CONTEXT:
{context}

ANSWER TO VERIFY:
{answer}

ORIGINAL QUESTION:
{question}

Analyze the answer and output JSON:
{{
    "all_claims_supported": true/false,
    "grounding_score": 0.0 to 1.0,
    "unsupported_claims": ["claim1", "claim2"],
    "missing_aspects": ["aspect1", "aspect2"],
    "suggested_queries": ["query1", "query2"],
    "reasoning": "Brief explanation"
}}

Rules:
- grounding_score: proportion of claims supported by context
- unsupported_claims: factual statements not found in context
- missing_aspects: parts of the question not addressed
- suggested_queries: queries to retrieve missing information

Output ONLY valid JSON:"""


COVERAGE_CHECK_PROMPT = """Check if this answer fully addresses the question.

QUESTION: {question}

ANSWER: {answer}

Does the answer address all parts of the question?
Output JSON:
{{
    "fully_addressed": true/false,
    "addressed_aspects": ["aspect1", "aspect2"],
    "missing_aspects": ["aspect1"],
    "coverage_ratio": 0.0 to 1.0
}}

Output ONLY valid JSON:"""


class VerificationAgentImpl(BaseAgent):
    """
    Verifies answer quality and triggers corrections.
    
    Checks:
    1. Citation coverage - are claims cited?
    2. Factual grounding - are claims supported by context?
    3. Query coverage - does answer address all parts of query?
    
    If verification fails, generates refined queries for re-retrieval.
    """
    
    name: str = "verification_agent"
    description: str = "Verifies answer quality and triggers corrections"
    
    def __init__(
        self,
        lm_client: LMStudioClient,
        min_grounding_score: float = 0.7,
        min_citation_coverage: float = 0.7,
        use_rule_based_precheck: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(lm_client, max_iterations=1, verbose=verbose)
        self.min_grounding_score = min_grounding_score
        self.min_citation_coverage = min_citation_coverage
        self.use_rule_based_precheck = use_rule_based_precheck
        
        # Lazy load verifiers
        self._citation_extractor = None
        self._rule_verifier = None
    
    @property
    def citation_extractor(self):
        """Lazy load citation extractor."""
        if self._citation_extractor is None:
            try:
                from app.services.citation_extractor import get_citation_extractor
                self._citation_extractor = get_citation_extractor()
            except ImportError:
                pass
        return self._citation_extractor
    
    @property
    def rule_verifier(self):
        """Lazy load rule-based verifier."""
        if self._rule_verifier is None:
            try:
                from app.services.answer_verifier import RuleBasedVerifier
                self._rule_verifier = RuleBasedVerifier()
            except ImportError:
                pass
        return self._rule_verifier
    
    async def run(self, state: AgentState) -> AgentState:
        """
        Verify the answer and determine if re-retrieval is needed.
        """
        logger.debug(f"[{self.name}] Starting verification")
        
        if not state.final_answer:
            logger.warning(f"[{self.name}] No answer to verify")
            state.verification_result = VerificationResult(
                passed=False,
                reason="No answer generated"
            )
            return state
        
        # Step 1: Quick rule-based check
        if self.use_rule_based_precheck:
            quick_result = self._quick_check(state)
            if quick_result.passed:
                logger.debug(f"[{self.name}] Passed quick check")
                state.verification_result = quick_result
                return state
        
        # Step 2: Check citations
        citation_result = self._check_citations(state)
        
        # Step 3: Check grounding with LLM
        grounding_result = await self._check_grounding(state)
        
        # Step 4: Check query coverage
        coverage_result = await self._check_coverage(state)
        
        # Step 5: Combine results
        verification = self._combine_results(
            citation_result,
            grounding_result,
            coverage_result,
            state
        )
        
        state.verification_result = verification
        
        logger.info(
            f"[{self.name}] Verification: passed={verification.passed}, "
            f"grounding={verification.grounding_score:.2f}, "
            f"citations={verification.citation_coverage:.2f}"
        )
        
        return state
    
    async def think(self, state: AgentState) -> str:
        """Analyze verification status."""
        if not state.final_answer:
            return "No answer to verify"
        
        if state.verification_result:
            if state.verification_result.passed:
                return "Verification passed"
            return f"Verification failed: {state.verification_result.reason}"
        
        return "Need to verify answer quality"
    
    async def act(self, state: AgentState, thought: str) -> AgentAction:
        """Decide verification action."""
        if "passed" in thought.lower():
            return AgentAction.stop("Verification passed")
        
        if "failed" in thought.lower():
            return AgentAction(
                type=ActionType.RE_RETRIEVE,
                reasoning="Re-retrieval needed due to verification failure"
            )
        
        return AgentAction(
            type=ActionType.VERIFY,
            reasoning="Verifying answer"
        )
    
    async def observe(self, state: AgentState, action: AgentAction) -> AgentState:
        """Process verification results."""
        if state.verification_result:
            status = "passed" if state.verification_result.passed else "failed"
            state.add_observation(f"Verification {status}")
        return state
    
    def _quick_check(self, state: AgentState) -> VerificationResult:
        """
        Fast rule-based verification without LLM.
        """
        if not self.rule_verifier:
            return VerificationResult(passed=False, reason="Rule verifier not available")
        
        try:
            grounded, confidence, issues = self.rule_verifier.quick_check(
                state.final_answer,
                state.retrieved_context
            )
            
            return VerificationResult(
                passed=grounded and confidence >= self.min_grounding_score,
                grounding_score=confidence,
                reason="; ".join(issues) if issues else "Quick check passed"
            )
        except Exception as e:
            logger.debug(f"Quick check failed: {e}")
            return VerificationResult(passed=False, reason=str(e))
    
    def _check_citations(self, state: AgentState) -> Tuple[bool, float, List[str]]:
        """
        Check citation coverage in the answer.
        
        Returns:
            Tuple of (passed, coverage_ratio, missing_citations)
        """
        if not self.citation_extractor or not state.retrieved_context:
            return True, 1.0, []
        
        try:
            passed, analysis = self.citation_extractor.passes_coverage_threshold(
                state.final_answer,
                state.retrieved_context
            )
            
            missing = []
            if hasattr(analysis, 'uncited_claims'):
                missing = analysis.uncited_claims[:3]
            
            return passed, analysis.coverage_ratio, missing
            
        except Exception as e:
            logger.debug(f"Citation check failed: {e}")
            return True, 1.0, []  # Fail open
    
    async def _check_grounding(self, state: AgentState) -> VerificationResult:
        """
        Check if answer claims are grounded in context using LLM.
        """
        if not state.retrieved_context:
            return VerificationResult(
                passed=False,
                grounding_score=0.0,
                reason="No context to verify against"
            )
        
        # Format context
        context_text = self._format_context(state.retrieved_context)
        
        prompt = VERIFICATION_PROMPT.format(
            context=context_text,
            answer=state.final_answer,
            question=state.query
        )
        
        try:
            response = await self._call_llm(prompt)
            return self._parse_grounding_response(response)
        except Exception as e:
            logger.warning(f"[{self.name}] Grounding check failed: {e}")
            return VerificationResult(
                passed=True,  # Fail open
                grounding_score=0.5,
                reason=f"Grounding check error: {str(e)}"
            )
    
    async def _check_coverage(self, state: AgentState) -> Tuple[bool, float, List[str]]:
        """
        Check if answer addresses all parts of the query.
        
        Returns:
            Tuple of (fully_addressed, coverage_ratio, missing_aspects)
        """
        prompt = COVERAGE_CHECK_PROMPT.format(
            question=state.query,
            answer=state.final_answer
        )
        
        try:
            response = await self._call_llm(prompt)
            
            # Parse response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return (
                    data.get("fully_addressed", True),
                    data.get("coverage_ratio", 1.0),
                    data.get("missing_aspects", [])
                )
        except Exception as e:
            logger.debug(f"Coverage check failed: {e}")
        
        return True, 1.0, []  # Fail open
    
    def _combine_results(
        self,
        citation_result: Tuple[bool, float, List[str]],
        grounding_result: VerificationResult,
        coverage_result: Tuple[bool, float, List[str]],
        state: AgentState,
    ) -> VerificationResult:
        """
        Combine all verification results into final decision.
        """
        citation_passed, citation_coverage, missing_citations = citation_result
        coverage_passed, coverage_ratio, missing_aspects = coverage_result
        
        # Overall pass criteria
        passed = (
            grounding_result.grounding_score >= self.min_grounding_score and
            citation_coverage >= self.min_citation_coverage and
            coverage_passed
        )
        
        # Build reason if failed
        reasons = []
        if grounding_result.grounding_score < self.min_grounding_score:
            reasons.append(f"Low grounding ({grounding_result.grounding_score:.1%})")
        if citation_coverage < self.min_citation_coverage:
            reasons.append(f"Low citation coverage ({citation_coverage:.1%})")
        if not coverage_passed:
            reasons.append("Question not fully addressed")
        
        # Generate refined queries if needed
        refined_queries = []
        if not passed:
            refined_queries = self._generate_refined_queries(
                state,
                grounding_result.unsupported_claims,
                missing_aspects
            )
        
        action = "none" if passed else "re_retrieve"
        
        return VerificationResult(
            passed=passed,
            action=action,
            refined_queries=refined_queries,
            reason="; ".join(reasons) if reasons else "All checks passed",
            citation_coverage=citation_coverage,
            grounding_score=grounding_result.grounding_score,
            unsupported_claims=grounding_result.unsupported_claims,
            answer=state.final_answer if passed else None
        )
    
    def _generate_refined_queries(
        self,
        state: AgentState,
        unsupported_claims: List[str],
        missing_aspects: List[str],
    ) -> List[str]:
        """
        Generate refined queries to address verification failures.
        """
        queries = []
        
        # Queries for unsupported claims
        for claim in unsupported_claims[:2]:
            # Extract key terms from claim
            queries.append(f"evidence for: {claim}")
        
        # Queries for missing aspects
        for aspect in missing_aspects[:2]:
            queries.append(f"{state.query} {aspect}")
        
        # If no specific queries, try query variations
        if not queries:
            queries.append(f"details about {state.query}")
        
        return queries[:3]  # Limit to 3 refined queries
    
    def _format_context(self, context: List[RetrievalHit]) -> str:
        """Format context for verification prompt."""
        parts = []
        for hit in context[:5]:  # Limit to top 5
            source = hit.metadata.get("source", hit.id)
            content = hit.content[:500] if len(hit.content) > 500 else hit.content
            parts.append(f"[{source}]:\n{content}")
        return "\n\n".join(parts)
    
    def _parse_grounding_response(self, response: str) -> VerificationResult:
        """Parse LLM grounding check response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                return VerificationResult(
                    passed=data.get("all_claims_supported", True),
                    grounding_score=data.get("grounding_score", 0.5),
                    unsupported_claims=data.get("unsupported_claims", []),
                    refined_queries=data.get("suggested_queries", []),
                    reason=data.get("reasoning", "")
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Failed to parse grounding response: {e}")
        
        # Default result
        return VerificationResult(
            passed=True,
            grounding_score=0.5,
            reason="Could not parse verification response"
        )


# Alias for backward compatibility
VerificationAgent = VerificationAgentImpl


def get_verification_agent(
    lm_client: Optional[LMStudioClient] = None,
) -> VerificationAgentImpl:
    """Factory function to create verification agent."""
    if lm_client is None:
        from app.clients.lmstudio import get_lmstudio_client
        lm_client = get_lmstudio_client()
    
    return VerificationAgentImpl(
        lm_client=lm_client,
        min_grounding_score=settings.rag.min_confidence_threshold,
        min_citation_coverage=settings.rag.min_citation_coverage,
    )
