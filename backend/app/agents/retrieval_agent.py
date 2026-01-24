"""
Retrieval Agent.

Implements adaptive multi-hop retrieval with strategy selection.
Decides when to retrieve more, when to expand knowledge graph,
and when sufficient context has been gathered.
"""

import logging
from typing import List, Optional, Set

from app.agents.base import BaseAgent
from app.agents.state import (
    AgentAction,
    AgentState,
    ActionType,
    CoverageAssessment,
    SubQuery,
)
from app.clients.lmstudio import LMStudioClient
from app.config import settings
from app.models.schemas import RetrievalHit
from app.rag.retriever import search_with_scopes

logger = logging.getLogger(__name__)


class RetrievalAgent(BaseAgent):
    """
    Adaptive retrieval agent with multi-hop capability.
    
    Features:
    - Strategy selection (hybrid, vector-only, BM25-only)
    - Coverage assessment to decide if more retrieval needed
    - Multi-hop retrieval following entity links
    - Knowledge graph expansion when appropriate
    - Client isolation enforcement
    """
    
    name: str = "retrieval_agent"
    description: str = "Adaptive multi-hop document retrieval"
    
    def __init__(
        self,
        lm_client: Optional[LMStudioClient] = None,
        max_hops: int = 2,
        min_coverage_threshold: float = 0.6,
        max_documents: int = 10,
        verbose: bool = False,
    ) -> None:
        super().__init__(lm_client, max_iterations=max_hops + 1, verbose=verbose)
        self.max_hops = max_hops
        self.min_coverage_threshold = min_coverage_threshold
        self.max_documents = max_documents
        
        # Lazy load dependencies
        self._retriever = None
        self._hybrid_search = None
        self._entity_extractor = None
    
    @property
    def retriever(self):
        """Lazy load retriever."""
        if self._retriever is None:
            from app.rag.retriever import get_retriever
            self._retriever = get_retriever()
        return self._retriever
    
    def _get_hybrid_search(self, client_id: str):
        """Get hybrid search for a specific client."""
        from app.rag.hybrid_search import get_hybrid_search
        return get_hybrid_search(client_id=client_id)
    
    @property
    def entity_extractor(self):
        """Lazy load entity extractor."""
        if self._entity_extractor is None:
            try:
                from app.knowledge.entity_extractor import get_entity_extractor
                self._entity_extractor = get_entity_extractor()
            except ImportError:
                logger.warning("Entity extractor not available")
        return self._entity_extractor
    
    async def run(self, state: AgentState) -> AgentState:
        """
        Execute retrieval for all sub-queries with multi-hop support.
        """
        logger.debug(f"[{self.name}] Starting retrieval")
        
        # Determine queries to retrieve for
        queries_to_process = []
        
        if state.sub_queries:
            # Process each unexecuted sub-query
            queries_to_process = [sq for sq in state.sub_queries if not sq.executed]
        else:
            # No sub-queries, use main query
            queries_to_process = [SubQuery(query=state.query, purpose="Main query")]
        
        # Process each query
        for sub_query in queries_to_process:
            await self._process_query(state, sub_query)
        
        # Assess overall coverage
        coverage = self._assess_coverage(state)
        state.add_thought(f"Coverage assessment: {coverage.coverage_ratio:.1%}, sufficient: {coverage.is_sufficient}")
        
        # If coverage is low and we haven't hit max hops, try to improve
        if not coverage.is_sufficient and state.iteration < self.max_hops:
            state = await self._improve_coverage(state, coverage)
        
        logger.info(f"[{self.name}] Retrieved {len(state.retrieved_context)} documents total")
        
        return state
    
    async def think(self, state: AgentState) -> str:
        """Decide retrieval strategy based on current state."""
        coverage = self._assess_coverage(state)
        
        if coverage.is_sufficient:
            return "STOP: Sufficient context retrieved"
        
        if coverage.missing_entities:
            entities = ", ".join(coverage.missing_entities[:3])
            return f"RETRIEVE_MORE: Need info about {entities}"
        
        if coverage.needs_kg_expansion:
            return "EXPAND_KG: Follow entity relationships"
        
        if not state.retrieved_context:
            return "RETRIEVE: Initial retrieval needed"
        
        return "RETRIEVE_MORE: Coverage below threshold"
    
    async def act(self, state: AgentState, thought: str) -> AgentAction:
        """Choose retrieval action based on thought."""
        if "STOP" in thought:
            return AgentAction.stop("Sufficient context")
        
        if "EXPAND_KG" in thought:
            return AgentAction(
                type=ActionType.EXPAND_KG,
                reasoning="Expanding knowledge graph"
            )
        
        if "RETRIEVE_MORE" in thought:
            return AgentAction(
                type=ActionType.RETRIEVE_MORE,
                reasoning=thought.split(": ", 1)[1] if ": " in thought else "Need more context"
            )
        
        return AgentAction(
            type=ActionType.RETRIEVE,
            params={"query": state.query},
            reasoning="Initial retrieval"
        )
    
    async def observe(self, state: AgentState, action: AgentAction) -> AgentState:
        """Process retrieval results."""
        count = len(state.retrieved_context)
        state.add_observation(f"Retrieved {count} documents")
        return state
    
    async def _process_query(self, state: AgentState, sub_query: SubQuery) -> None:
        """
        Process a single query with the appropriate retrieval strategy.
        """
        logger.debug(f"[{self.name}] Processing: {sub_query.query[:50]}...")
        
        client_id = state.client_id or "global"
        
        try:
            hits = search_with_scopes(
                query=sub_query.query,
                client_id=client_id,
                top_k=settings.rag.rerank_top_k,
                fetch_k=settings.rag.initial_fetch_k,
                allowed_clients=state.allowed_clients,
            )
            
            # Store results
            sub_query.results = hits
            sub_query.executed = True
            
            # Merge into overall context
            state.merge_retrieved(hits, sub_query.query)
            
            logger.debug(f"[{self.name}] Retrieved {len(hits)} docs for sub-query")
            
        except Exception as e:
            logger.error(f"[{self.name}] Retrieval failed: {e}")
            sub_query.executed = True  # Mark as executed even on failure
    
    def _assess_coverage(self, state: AgentState) -> CoverageAssessment:
        """
        Assess how well the retrieved context covers the query.
        """
        if not state.retrieved_context:
            return CoverageAssessment(
                is_sufficient=False,
                coverage_ratio=0.0,
                confidence=0.0,
                notes="No documents retrieved"
            )
        
        # Basic coverage heuristics
        num_docs = len(state.retrieved_context)
        avg_score = sum(h.score for h in state.retrieved_context) / num_docs
        
        # Check if top results have good scores
        # Note: For distance metrics, lower is better, for similarity, higher is better
        # We normalize to assume higher = better
        top_scores = [h.score for h in state.retrieved_context[:3]]
        top_avg = sum(top_scores) / len(top_scores) if top_scores else 0
        
        # Determine if using distance or similarity metric
        # Distance metrics typically have values > 0.3 for poor matches
        # Similarity metrics have values < 0.5 for poor matches
        is_distance_metric = top_avg > 0.5 and all(s > 0 for s in top_scores)
        
        if is_distance_metric:
            # Lower is better for distance
            coverage_ratio = max(0, 1 - top_avg)
        else:
            # Higher is better for similarity (but scores might be 0-1 or higher)
            coverage_ratio = min(1.0, top_avg)
        
        # Check for entity coverage
        missing_entities = self._find_missing_entities(state)
        
        # Determine if we need KG expansion
        needs_kg = (
            coverage_ratio < 0.5 and 
            settings.rag.knowledge_graph_enabled and
            len(missing_entities) > 0
        )
        
        is_sufficient = (
            coverage_ratio >= self.min_coverage_threshold and
            num_docs >= 2 and
            len(missing_entities) == 0
        )
        
        return CoverageAssessment(
            is_sufficient=is_sufficient,
            coverage_ratio=coverage_ratio,
            missing_entities=missing_entities,
            needs_kg_expansion=needs_kg,
            confidence=coverage_ratio,
            notes=f"{num_docs} docs, avg_score={avg_score:.3f}"
        )
    
    def _find_missing_entities(self, state: AgentState) -> List[str]:
        """Find entities mentioned in query but not in retrieved context."""
        if not self.entity_extractor:
            return []
        
        try:
            # Extract entities from query
            query_entities = self.entity_extractor.extract_from_text(state.query)
            
            # Extract entities from context
            context_text = " ".join(h.content for h in state.retrieved_context)
            context_entities = self.entity_extractor.extract_from_text(context_text)
            
            # Find missing
            query_set = {e.lower() for e in query_entities}
            context_set = {e.lower() for e in context_entities}
            
            missing = query_set - context_set
            return list(missing)[:5]  # Limit to top 5
            
        except Exception as e:
            logger.debug(f"Entity extraction failed: {e}")
            return []
    
    async def _improve_coverage(
        self, 
        state: AgentState, 
        coverage: CoverageAssessment
    ) -> AgentState:
        """
        Attempt to improve coverage through additional retrieval.
        """
        logger.debug(f"[{self.name}] Attempting to improve coverage")
        
        if coverage.needs_kg_expansion:
            state = await self._expand_knowledge_graph(state)
        
        elif coverage.missing_entities:
            # Create additional queries for missing entities
            for entity in coverage.missing_entities[:2]:  # Limit additional queries
                additional_query = f"{entity} {state.query}"
                sub_query = SubQuery(
                    query=additional_query,
                    purpose=f"Additional retrieval for entity: {entity}"
                )
                await self._process_query(state, sub_query)
                state.sub_queries.append(sub_query)
        
        return state
    
    async def _expand_knowledge_graph(self, state: AgentState) -> AgentState:
        """
        Expand retrieval using knowledge graph relationships.
        """
        if not state.client_id:
            return state
        
        try:
            from app.knowledge.graph_query import get_graph_query_expander
            
            kg_expander = get_graph_query_expander()
            _, expansion = kg_expander.enrich_retrieval(
                state.query, 
                state.retrieved_context, 
                state.client_id
            )
            
            if expansion.identified_entities:
                state.add_observation(
                    f"KG expansion found {len(expansion.identified_entities)} entities, "
                    f"{len(expansion.related_entities)} related"
                )
            
            logger.debug(f"[{self.name}] KG expansion complete")
            
        except Exception as e:
            logger.warning(f"[{self.name}] KG expansion failed: {e}")
        
        return state
    
    async def multi_hop_retrieve(
        self,
        query: str,
        state: AgentState,
        max_hops: Optional[int] = None,
    ) -> List[RetrievalHit]:
        """
        Chain multiple retrievals following entity links.
        
        This method performs iterative retrieval where each hop:
        1. Retrieves documents for current query
        2. Extracts new entities from results
        3. Forms a follow-up query from discovered entities
        4. Continues until coverage is sufficient or max hops reached
        
        Args:
            query: Initial query
            state: Agent state for client isolation
            max_hops: Maximum retrieval hops (default: self.max_hops)
            
        Returns:
            Combined list of retrieval hits from all hops
        """
        max_hops = max_hops or self.max_hops
        all_hits: List[RetrievalHit] = []
        seen_ids: Set[str] = set()
        current_query = query
        
        for hop in range(max_hops):
            logger.debug(f"[{self.name}] Multi-hop retrieval: hop {hop + 1}/{max_hops}")
            
            # Retrieve for current query
            sub_query = SubQuery(query=current_query, purpose=f"Hop {hop + 1}")
            await self._process_query(state, sub_query)
            
            # Add new hits
            new_hits = [h for h in sub_query.results if h.id not in seen_ids]
            for hit in new_hits:
                all_hits.append(hit)
                seen_ids.add(hit.id)
            
            # Check if we have enough
            if len(all_hits) >= self.max_documents:
                break
            
            # Extract entities for next hop
            if self.entity_extractor and new_hits:
                context_text = " ".join(h.content for h in new_hits)
                entities = self.entity_extractor.extract_from_text(context_text)
                
                if not entities:
                    break
                
                # Form follow-up query
                top_entities = entities[:3]
                current_query = f"{' '.join(top_entities)} related to {query}"
            else:
                break
        
        # Sort by score and return
        all_hits.sort(key=lambda h: h.score, reverse=True)
        return all_hits[:self.max_documents]


def get_retrieval_agent(
    lm_client: Optional[LMStudioClient] = None,
) -> RetrievalAgent:
    """Factory function to create retrieval agent."""
    return RetrievalAgent(
        lm_client=lm_client,
        max_hops=2,
        min_coverage_threshold=0.6,
    )
