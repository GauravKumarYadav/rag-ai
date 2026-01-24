"""
Query Decomposition Agent.

Breaks complex queries into independent sub-queries that can be
answered separately and then synthesized into a final response.
"""

import json
import logging
import re
from typing import List, Optional

from app.agents.base import BaseAgent
from app.agents.state import AgentAction, AgentState, ActionType, SubQuery
from app.clients.lmstudio import LMStudioClient
from app.config import settings

logger = logging.getLogger(__name__)


DECOMPOSITION_PROMPT = """Break this query into independent sub-queries that can be answered separately.

Query: {query}

Conversation context: {context}

Rules:
1. Each sub-query should be self-contained and answerable independently
2. Maximum {max_sub_queries} sub-queries (to respect token budget)
3. If the query is simple and doesn't need decomposition, return a single sub-query
4. Preserve entity names and specific terms exactly as stated
5. For comparison queries (X vs Y), create separate queries for each item
6. For temporal queries (before/after), identify the time periods

Examples:
- "Compare the revenue of company A and company B" -> 
  ["What is the revenue of company A?", "What is the revenue of company B?"]
- "How did sales change from Q1 to Q2 2024?" ->
  ["What were the sales in Q1 2024?", "What were the sales in Q2 2024?"]
- "What is the capital of France?" -> 
  ["What is the capital of France?"] (simple, no decomposition needed)

Output ONLY valid JSON:
{{
    "sub_queries": [
        {{
            "query": "the sub-query text",
            "purpose": "why this sub-query is needed",
            "retrieval_hints": ["keyword1", "keyword2"]
        }}
    ],
    "reasoning": "Brief explanation of the decomposition"
}}"""


COMPLEXITY_INDICATORS = [
    # Comparison patterns
    r'\b(compare|versus|vs\.?|difference between|similarities)\b',
    r'\b(better|worse|more|less) than\b',
    
    # Temporal patterns
    r'\b(before|after|between|from .* to|change|trend|over time)\b',
    r'\b(Q[1-4]|quarter|year|month|2\d{3})\b',
    
    # Multi-part patterns
    r'\b(and|also|additionally|furthermore)\b.*\?',
    r'\b(both|each|all|every)\b',
    
    # Conditional patterns
    r'\b(if|when|unless|assuming)\b',
    
    # Enumeration patterns
    r'\b(list|enumerate|what are the|how many)\b',
]


class QueryDecomposer(BaseAgent):
    """
    Decomposes complex queries into retrievable sub-queries.
    
    Uses complexity detection to decide whether decomposition is needed,
    and respects token budget constraints by limiting sub-query count.
    """
    
    name: str = "query_decomposer"
    description: str = "Breaks complex queries into sub-queries"
    
    def __init__(
        self,
        lm_client: LMStudioClient,
        max_sub_queries: int = 3,
        use_complexity_detection: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(lm_client, max_iterations=1, verbose=verbose)
        self.max_sub_queries = max_sub_queries
        self.use_complexity_detection = use_complexity_detection
        
        # Compile complexity patterns
        self._complexity_patterns = [
            re.compile(p, re.IGNORECASE) for p in COMPLEXITY_INDICATORS
        ]
    
    async def run(self, state: AgentState) -> AgentState:
        """
        Decompose the query if it's complex enough to warrant it.
        """
        logger.debug(f"[{self.name}] Analyzing query for decomposition")
        
        # Check if decomposition is needed
        if self.use_complexity_detection:
            complexity_score = self._assess_complexity(state.query)
            
            if complexity_score < 0.3:
                # Simple query - no decomposition needed
                logger.debug(f"[{self.name}] Simple query (score: {complexity_score:.2f}), skipping decomposition")
                state.sub_queries = [SubQuery(
                    query=state.query,
                    purpose="Direct retrieval",
                    executed=False
                )]
                state.add_thought(f"Query is simple enough for direct retrieval (complexity: {complexity_score:.2f})")
                return state
        
        # Decompose the query using LLM
        try:
            sub_queries = await self._decompose(state)
            state.sub_queries = sub_queries
            
            # Log decomposition
            if len(sub_queries) > 1:
                queries_str = "; ".join([sq.query for sq in sub_queries])
                state.add_thought(f"Decomposed into {len(sub_queries)} sub-queries: {queries_str}")
                logger.info(f"[{self.name}] Decomposed into {len(sub_queries)} sub-queries")
            else:
                state.add_thought("No decomposition needed - single query sufficient")
            
        except Exception as e:
            logger.warning(f"[{self.name}] Decomposition failed: {e}, using original query")
            state.sub_queries = [SubQuery(
                query=state.query,
                purpose="Direct retrieval (decomposition failed)",
                executed=False
            )]
        
        return state
    
    async def think(self, state: AgentState) -> str:
        """Analyze if decomposition is needed."""
        complexity = self._assess_complexity(state.query)
        
        if complexity >= 0.5:
            return f"Query appears complex (score: {complexity:.2f}), decomposition recommended"
        elif complexity >= 0.3:
            return f"Query has moderate complexity (score: {complexity:.2f}), decomposition may help"
        else:
            return f"Query is simple (score: {complexity:.2f}), direct retrieval sufficient"
    
    async def act(self, state: AgentState, thought: str) -> AgentAction:
        """Decide whether to decompose."""
        if "complex" in thought.lower() or "moderate" in thought.lower():
            return AgentAction(
                type=ActionType.DECOMPOSE,
                reasoning="Decomposing complex query"
            )
        return AgentAction.stop("Query is simple, no decomposition needed")
    
    async def observe(self, state: AgentState, action: AgentAction) -> AgentState:
        """Process decomposition results."""
        state.add_observation(f"Decomposed into {len(state.sub_queries)} sub-queries")
        return state
    
    def _assess_complexity(self, query: str) -> float:
        """
        Assess query complexity using pattern matching.
        
        Returns a score between 0 (simple) and 1 (complex).
        """
        score = 0.0
        matches = 0
        
        for pattern in self._complexity_patterns:
            if pattern.search(query):
                matches += 1
        
        # Base score from pattern matches
        if matches > 0:
            score = min(1.0, matches * 0.25)
        
        # Adjust based on query length
        words = len(query.split())
        if words > 20:
            score += 0.2
        if words > 30:
            score += 0.1
        
        # Check for question words indicating multi-part
        if re.search(r'\b(what|who|where|when|why|how)\b.*\b(and|also)\b', query, re.I):
            score += 0.2
        
        # Check for explicit enumeration
        if re.search(r'\b(list|all|each|every)\b', query, re.I):
            score += 0.15
        
        return min(1.0, score)
    
    async def _decompose(self, state: AgentState) -> List[SubQuery]:
        """Decompose query using LLM."""
        context = state.get_context_summary()
        
        prompt = DECOMPOSITION_PROMPT.format(
            query=state.query,
            context=context,
            max_sub_queries=self.max_sub_queries
        )
        
        response = await self._call_llm(prompt)
        return self._parse_decomposition(response, state.query)
    
    def _parse_decomposition(self, response: str, original_query: str) -> List[SubQuery]:
        """Parse LLM response into SubQuery objects."""
        try:
            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                sub_queries = []
                for sq_data in data.get("sub_queries", [])[:self.max_sub_queries]:
                    sub_queries.append(SubQuery(
                        query=sq_data.get("query", original_query),
                        purpose=sq_data.get("purpose", ""),
                        retrieval_hints=sq_data.get("retrieval_hints", []),
                        executed=False
                    ))
                
                if sub_queries:
                    return sub_queries
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse decomposition: {e}")
        
        # Fallback: return original query as single sub-query
        return [SubQuery(
            query=original_query,
            purpose="Direct retrieval",
            executed=False
        )]
    
    def allocate_token_budget(
        self,
        sub_queries: List[SubQuery],
        total_budget: Optional[int] = None,
    ) -> dict[str, int]:
        """
        Allocate token budget across sub-queries.
        
        Divides the total context token budget proportionally
        across sub-queries, ensuring each gets a minimum allocation.
        
        Args:
            sub_queries: List of sub-queries to allocate budget for
            total_budget: Total token budget (uses config default if None)
            
        Returns:
            Dictionary mapping sub-query text to allocated tokens
        """
        total_budget = total_budget or settings.rag.context_token_budget
        
        if not sub_queries:
            return {}
        
        # Reserve some budget for synthesis overhead
        usable_budget = int(total_budget * 0.9)
        min_per_query = 100  # Minimum useful allocation
        
        num_queries = len(sub_queries)
        per_query = max(min_per_query, usable_budget // num_queries)
        
        return {sq.query: per_query for sq in sub_queries}


def get_query_decomposer(
    lm_client: Optional[LMStudioClient] = None,
) -> QueryDecomposer:
    """Factory function to create query decomposer."""
    if lm_client is None:
        from app.clients.lmstudio import get_lmstudio_client
        lm_client = get_lmstudio_client()
    
    # Get max sub-queries from config if available
    max_sub_queries = 3
    try:
        from app.config import settings
        if hasattr(settings, 'agent') and hasattr(settings.agent, 'max_sub_queries'):
            max_sub_queries = settings.agent.max_sub_queries
    except (ImportError, AttributeError):
        pass
    
    return QueryDecomposer(
        lm_client=lm_client,
        max_sub_queries=max_sub_queries,
    )
