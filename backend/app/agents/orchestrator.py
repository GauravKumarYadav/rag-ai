"""
Orchestrator Agent.

The main coordinator that analyzes queries and routes them to appropriate
specialized agents. Manages the overall execution flow and combines results.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from app.agents.base import BaseAgent
from app.agents.state import (
    AgentAction,
    AgentPlan,
    AgentState,
    ActionType,
    SubQuery,
    VerificationResult,
)
from app.clients.lmstudio import LMStudioClient
from app.models.schemas import RetrievalHit
from app.services.document_listing import list_documents
from app.services.natural_responses import (
    format_tool_results_response,
    get_conversational_response,
)
from app.services.response_generator import generate_response
from app.services.query_processor import QueryProcessor, Intent, get_query_processor
from app.memory.conversation_state import ConversationState

if TYPE_CHECKING:
    from app.agents.query_decomposer import QueryDecomposer
    from app.agents.retrieval_agent import RetrievalAgent
    from app.agents.synthesis_agent import SynthesisAgent
    from app.agents.verification_agent import VerificationAgentImpl
    from app.agents.tool_agent import ToolAgent

logger = logging.getLogger(__name__)


# Patterns for rejection/negative responses
REJECTION_PATTERNS = [
    r"^no[\s,!.]*$",
    r"^(not|nope|wrong|incorrect|that'?s? not)[\s\w]*$",
    r"^(try again|different|another)[\s\w]*$",
]


PLANNING_PROMPT = """You are a query analyzer. Analyze this query and create an execution plan.

Query: {query}

Conversation context: {context}

Available capabilities:
- query_decomposer: For complex multi-part questions that need to be broken down
- retrieval_agent: For searching documents and retrieving relevant information
- knowledge_graph_agent: For questions about entity relationships
- tool_agent: For calculations, date operations, or data queries
- synthesis_agent: For combining multiple pieces of information

Analyze the query and output a JSON plan:
{{
    "complexity": "simple|moderate|complex",
    "needs_decomposition": true/false,
    "needs_tools": true/false,
    "tool_names": ["calculator", "datetime"] or [],
    "agent_sequence": ["agent1", "agent2"],
    "reasoning": "Brief explanation of the plan"
}}

Rules:
- "simple": Direct question, single retrieval sufficient
- "moderate": May need multiple retrievals or a tool
- "complex": Multi-part question, needs decomposition
- Most questions are "simple" - don't over-complicate
- Only decompose if query has multiple distinct parts (e.g., "Compare X and Y")
- Use tools only for calculations, dates, or explicit data operations

Output ONLY valid JSON:"""


class OrchestratorAgent(BaseAgent):
    """
    Routes queries to appropriate sub-agents and coordinates execution.
    
    The orchestrator:
    1. Analyzes the query to determine complexity
    2. Creates an execution plan
    3. Routes to specialized agents
    4. Manages the correction loop if verification fails
    5. Combines results into a final response
    """
    
    name: str = "orchestrator"
    description: str = "Main coordinator for agentic RAG pipeline"
    
    def __init__(
        self,
        lm_client: LMStudioClient,
        retrieval_agent: Optional["RetrievalAgent"] = None,
        query_decomposer: Optional["QueryDecomposer"] = None,
        synthesis_agent: Optional["SynthesisAgent"] = None,
        verification_agent: Optional["VerificationAgentImpl"] = None,
        tool_agent: Optional["ToolAgent"] = None,
        max_iterations: int = 3,
        max_corrections: int = 2,
        verbose: bool = False,
    ) -> None:
        super().__init__(lm_client, max_iterations, verbose)
        
        self.retrieval_agent = retrieval_agent
        self.query_decomposer = query_decomposer
        self.synthesis_agent = synthesis_agent
        self.verification_agent = verification_agent
        self.tool_agent = tool_agent
        self.max_corrections = max_corrections
        self._query_processor = get_query_processor()
    
    async def run(self, state: AgentState) -> AgentState:
        """
        Execute the full orchestrated pipeline with intent gating.
        
        Flow:
        0. Classify intent - gate processing based on intent type
        1. Plan: Analyze query and create execution plan
        2. Decompose: Break into sub-queries if needed
        3. Retrieve: Get relevant documents for each sub-query
        4. Tools: Execute any required tools
        5. Synthesize: Generate answer from context
        6. Verify: Check answer quality
        7. Correct: Re-retrieve if verification fails
        """
        state.started_at = time.time()
        state.original_query = state.query
        
        logger.info(f"[{self.name}] Starting orchestration for: {state.query[:100]}...")
        
        try:
            # Step 0: Classify intent FIRST
            state = await self._classify_intent(state)
            
            # Step 0.5: Intent-based early exit
            if state.intent == Intent.DOCUMENT_LIST.value:
                documents = list_documents(client_id=state.client_id)
                state.final_answer = await self._generate_llm_response(
                    response_type="document_list",
                    user_message=state.query,
                    documents=documents,
                )
                state.add_thought("Document list intent - returned document listing")
                return state

            # Handle CHITCHAT - no retrieval needed
            if state.intent == Intent.CHITCHAT.value:
                # Check if this is actually a mixed intent (e.g., "Hi, what's the revenue?")
                if self._has_question_component(state.query):
                    logger.info(f"[{self.name}] Mixed intent detected, proceeding with RAG")
                    state.intent = Intent.QUESTION.value
                    state.needs_retrieval = True
                # Check if this is a rejection response
                elif self._is_rejection_response(state.query):
                    state.final_answer = await self._generate_llm_response(
                        response_type="rejection",
                        user_message=state.query,
                    )
                    state.add_thought("Detected rejection response, asking for clarification")
                    return state
                else:
                    state.final_answer = await self._generate_llm_response(
                        response_type="chitchat",
                        user_message=state.query,
                    )
                    state.add_thought("Chitchat intent - skipped retrieval")
                    return state
            
            # Handle CLARIFICATION - use conversation state only
            if state.intent == Intent.CLARIFICATION.value:
                state.final_answer = await self._generate_from_state(state)
                state.add_thought("Clarification intent - used conversation state only")
                return state
            
            # Handle FOLLOW_UP with no resolvable references
            if state.intent == Intent.FOLLOW_UP.value and not state.resolved_references:
                # Check if there's any conversation history
                conv_state = state.conversation_state
                if not conv_state or not getattr(conv_state, 'entities', None):
                    state.final_answer = await self._generate_llm_response(
                        response_type="followup_needs_reference",
                        user_message=state.query,
                    )
                    state.add_thought("Follow-up with no references and no conversation history")
                    return state
                # Try to generate from state
                state.final_answer = await self._generate_from_state(state)
                state.add_thought("Follow-up intent - used conversation state")
                return state

            # Handle ACTION_REQUEST without retrieval
            if state.intent == Intent.ACTION_REQUEST.value and not state.needs_retrieval:
                state.final_answer = await self._generate_llm_response(
                    response_type="action_request",
                    user_message=state.query,
                )
                state.add_thought("Action request without retrieval - requested clarification")
                return state
            
            # Check if this is a tool-only query (calculations, dates)
            if self._is_tool_only_query(state):
                logger.info(f"[{self.name}] Tool-only query detected")
                # Set up plan for tool execution
                state.plan = AgentPlan(
                    estimated_complexity="simple",
                    requires_tools=True,
                    tool_names=["calculator", "datetime"],
                    agent_sequence=["tool_agent"],
                    reasoning="Tool-only query"
                )
                state = await self._execute_tools(state)
                if state.tool_results:
                    state.final_answer = await self._generate_llm_response(
                        response_type="tool_results",
                        user_message=state.query,
                        tool_results=state.tool_results,
                    )
                    state.add_thought("Tool-only query - returned tool results")
                    return state
                state.final_answer = await self._generate_llm_response(
                    response_type="tool_results",
                    user_message=state.query,
                    tool_results={},
                    needs_details=True,
                )
                state.add_thought("Tool-only query - missing parameters")
                return state
            
            # Continue with normal RAG pipeline for QUESTION, ACTION_REQUEST, 
            # or FOLLOW_UP with resolved references
            
            # Step 1: Create execution plan
            state = await self._plan(state)
            
            # Step 2: Decompose if needed
            if state.plan and state.plan.sub_queries:
                state = await self._execute_decomposition(state)
            
            # Step 3: Execute retrieval
            state = await self._execute_retrieval(state)
            
            # Step 4: Execute tools if needed
            if state.plan and state.plan.requires_tools:
                state = await self._execute_tools(state)
            
            # Step 5-7: Synthesize with correction loop
            state = await self._synthesize_with_correction(state)
            
        except Exception as e:
            logger.error(f"[{self.name}] Orchestration error: {e}")
            state.add_observation(f"Orchestration error: {str(e)}")
            # Attempt graceful degradation
            if not state.final_answer:
                state.final_answer = "I apologize, but I encountered an issue processing your request. Please try rephrasing your question."
        
        elapsed = time.time() - state.started_at
        logger.info(f"[{self.name}] Completed in {elapsed:.2f}s")
        
        return state
    
    async def think(self, state: AgentState) -> str:
        """Analyze what the orchestrator should do next."""
        if not state.plan:
            return "Need to analyze query and create execution plan"
        
        if state.plan.sub_queries and not all(sq.executed for sq in state.plan.sub_queries):
            pending = [sq.query for sq in state.plan.sub_queries if not sq.executed]
            return f"Need to execute {len(pending)} pending sub-queries"
        
        if not state.retrieved_context and not state.tool_results:
            return "Need to retrieve context or execute tools"
        
        if not state.final_answer:
            return "Have context, need to synthesize answer"
        
        if state.verification_result and not state.verification_result.passed:
            return f"Verification failed: {state.verification_result.reason}. Need to re-retrieve."
        
        return "Orchestration complete"
    
    async def act(self, state: AgentState, thought: str) -> AgentAction:
        """Choose the next action based on current state."""
        if "create execution plan" in thought.lower():
            return AgentAction(
                type=ActionType.DECOMPOSE,
                reasoning="Creating execution plan"
            )
        
        if "sub-queries" in thought.lower():
            return AgentAction(
                type=ActionType.RETRIEVE,
                reasoning="Executing sub-queries"
            )
        
        if "retrieve context" in thought.lower():
            return AgentAction(
                type=ActionType.RETRIEVE,
                params={"query": state.query},
                reasoning="Retrieving documents"
            )
        
        if "synthesize" in thought.lower():
            return AgentAction(
                type=ActionType.SYNTHESIZE,
                reasoning="Generating answer"
            )
        
        if "re-retrieve" in thought.lower():
            return AgentAction(
                type=ActionType.RE_RETRIEVE,
                reasoning="Re-retrieving due to verification failure"
            )
        
        return AgentAction.stop("Orchestration complete")
    
    async def observe(self, state: AgentState, action: AgentAction) -> AgentState:
        """Process action results."""
        state.add_observation(f"Executed {action.type.value}")
        return state
    
    async def _plan(self, state: AgentState) -> AgentState:
        """Create an execution plan for the query."""
        logger.debug(f"[{self.name}] Creating execution plan")
        
        context_summary = state.get_context_summary()
        
        prompt = PLANNING_PROMPT.format(
            query=state.query,
            context=context_summary
        )
        
        try:
            response = await self._call_llm(prompt)
            plan = self._parse_plan(response)
            state.plan = plan
            state.add_thought(f"Plan: {plan.reasoning}")
            logger.debug(f"[{self.name}] Plan complexity: {plan.estimated_complexity}")
        except Exception as e:
            logger.warning(f"[{self.name}] Planning failed, using simple plan: {e}")
            state.plan = AgentPlan(
                estimated_complexity="simple",
                agent_sequence=["retrieval_agent", "synthesis_agent"],
                reasoning="Default simple plan"
            )
        
        return state
    
    def _parse_plan(self, response: str) -> AgentPlan:
        """Parse LLM response into an AgentPlan."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                sub_queries = []
                if data.get("needs_decomposition"):
                    # Sub-queries will be generated by decomposer
                    pass
                
                return AgentPlan(
                    sub_queries=sub_queries,
                    agent_sequence=data.get("agent_sequence", ["retrieval_agent", "synthesis_agent"]),
                    requires_tools=data.get("needs_tools", False),
                    tool_names=data.get("tool_names", []),
                    estimated_complexity=data.get("complexity", "simple"),
                    reasoning=data.get("reasoning", ""),
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse plan: {e}")
        
        # Default plan
        return AgentPlan(
            estimated_complexity="simple",
            agent_sequence=["retrieval_agent", "synthesis_agent"],
            reasoning="Default plan (parsing failed)"
        )
    
    async def _classify_intent(self, state: AgentState) -> AgentState:
        """
        Classify query intent before processing.
        
        Uses QueryProcessor to determine if retrieval is needed
        and resolve any references to previous context.
        """
        # Get or create conversation state for reference resolution
        conv_state = state.conversation_state
        if conv_state is None:
            conv_state = ConversationState()
        
        # Process the query
        query_result = await self._query_processor.process(
            message=state.query,
            state=conv_state,
            lm_client=self.lm_client,
        )
        
        # Update state with intent info
        state.intent = query_result.intent.value
        state.intent_confidence = query_result.confidence
        state.needs_retrieval = query_result.needs_retrieval
        state.resolved_references = query_result.resolved_references
        
        # Update search query if rewritten
        if query_result.search_query != query_result.original_query:
            state.search_query = query_result.search_query
        else:
            state.search_query = state.query
        
        logger.info(
            f"[{self.name}] Intent: {state.intent}, confidence: {state.intent_confidence:.2f}, "
            f"needs_retrieval: {state.needs_retrieval}"
        )
        
        return state
    
    def _has_question_component(self, query: str) -> bool:
        """Check if a chitchat query also contains a question."""
        # Import retrieval keywords from query processor
        from app.services.query_processor import RETRIEVAL_KEYWORDS
        
        # Remove greeting prefix
        query_stripped = re.sub(r"^(hi|hello|hey)[,!.]?\s*", "", query, flags=re.I)
        
        # Check if remainder looks like a question
        if len(query_stripped) > 10:
            query_lower = query_stripped.lower()
            if any(kw in query_lower for kw in RETRIEVAL_KEYWORDS):
                return True
            if "?" in query_stripped:
                return True
        return False
    
    def _is_rejection_response(self, query: str) -> bool:
        """Check if query is a negative/rejection response."""
        for pattern in REJECTION_PATTERNS:
            if re.match(pattern, query.strip(), re.I):
                return True
        return False
    
    async def _generate_conversational_response(self, state: AgentState) -> str:
        """Generate response for chitchat without retrieval."""
        return get_conversational_response(state.query)

    async def _generate_llm_response(
        self,
        *,
        response_type: str,
        user_message: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[Dict[str, Any]] = None,
        needs_details: bool = False,
    ) -> str:
        """Generate an LLM-based response with structured inputs."""
        return await generate_response(
            lm_client=self.lm_client,
            response_type=response_type,
            user_message=user_message,
            documents=documents,
            tool_results=tool_results,
            needs_details=needs_details,
            allow_fallback_templates=False,
        )
    
    async def _generate_from_state(self, state: AgentState) -> str:
        """Generate response using only conversation state (no new retrieval)."""
        conv_state = state.conversation_state
        
        if not conv_state or not getattr(conv_state, 'running_summary', None):
            return await self._generate_llm_response(
                response_type="clarification",
                user_message=state.query,
            )
        
        # Use LLM to generate response from state context
        prompt = f"""Based on our conversation so far:
{conv_state.running_summary}

The user asked: {state.query}

Provide a helpful response based on the conversation context. If you don't have enough information, say so clearly."""

        try:
            response = await self._call_llm(prompt)
            return response if response else "Could you clarify what you're referring to?"
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to generate from state: {e}")
            return clarification_needs_context_response()
    
    def _is_tool_only_query(self, state: AgentState) -> bool:
        """Check if query only needs tools, not document retrieval."""
        query_lower = state.query.lower()
        
        # Calculation patterns
        calc_patterns = [
            r"\b(calculate|compute|what is|how much is)\s+[\d\+\-\*\/\(\)\.\s]+",
            r"\b\d+\s*[\+\-\*\/]\s*\d+",
            r"\b(add|subtract|multiply|divide)\s+\d+",
        ]
        
        # Date/time patterns
        date_patterns = [
            r"\b(what|current|today'?s?)\s+(date|time|day|month|year)\b",
            r"\b(days?|weeks?|months?|years?)\s+(between|from|until|since)\b",
            r"\bhow (long|many days)\b",
        ]
        
        for pattern in calc_patterns + date_patterns:
            if re.search(pattern, query_lower, re.I):
                return True
        
        return False
    
    def _format_tool_results_for_answer(self, state: AgentState) -> str:
        """Format tool results into a readable response."""
        return format_tool_results_response(state.tool_results)
    
    async def _execute_decomposition(self, state: AgentState) -> AgentState:
        """Execute query decomposition if needed."""
        if not self.query_decomposer:
            logger.warning(f"[{self.name}] Query decomposer not available")
            return state
        
        if state.plan.estimated_complexity == "simple":
            # Skip decomposition for simple queries
            return state
        
        logger.debug(f"[{self.name}] Executing query decomposition")
        state = await self.query_decomposer.run(state)
        
        return state
    
    async def _execute_retrieval(self, state: AgentState) -> AgentState:
        """Execute retrieval for the query/sub-queries."""
        if not self.retrieval_agent:
            logger.warning(f"[{self.name}] Retrieval agent not available")
            return state
        
        logger.debug(f"[{self.name}] Executing retrieval")
        state = await self.retrieval_agent.run(state)
        
        return state
    
    async def _execute_tools(self, state: AgentState) -> AgentState:
        """Execute required tools."""
        if not self.tool_agent:
            logger.warning(f"[{self.name}] Tool agent not available")
            return state
        
        if not state.plan or not state.plan.tool_names:
            return state
        
        logger.debug(f"[{self.name}] Executing tools: {state.plan.tool_names}")
        state = await self.tool_agent.run(state)
        
        return state
    
    async def _synthesize_with_correction(self, state: AgentState) -> AgentState:
        """
        Synthesize answer with self-correction loop.
        
        If verification fails, refine queries and re-retrieve.
        """
        for attempt in range(self.max_corrections + 1):
            # Synthesize answer
            if self.synthesis_agent:
                state = await self.synthesis_agent.run(state)
            else:
                # Fallback: simple synthesis
                state = await self._simple_synthesis(state)
            
            # Skip verification if disabled or no context
            if not self.verification_agent or not state.retrieved_context:
                break
            
            # Verify answer
            state = await self.verification_agent.run(state)
            
            if state.verification_result and state.verification_result.passed:
                logger.debug(f"[{self.name}] Verification passed on attempt {attempt + 1}")
                break
            
            if attempt < self.max_corrections:
                logger.info(f"[{self.name}] Verification failed, attempting correction {attempt + 1}")
                state.increment_correction()
                
                # Use refined queries from verification
                if state.verification_result and state.verification_result.refined_queries:
                    # Re-retrieve with refined queries
                    for refined_query in state.verification_result.refined_queries:
                        state.sub_queries.append(SubQuery(
                            query=refined_query,
                            purpose="Correction re-retrieval"
                        ))
                    
                    if self.retrieval_agent:
                        state = await self.retrieval_agent.run(state)
                
                state.add_thought(f"Correction attempt {attempt + 1}: {state.verification_result.reason if state.verification_result else 'unknown'}")
        
        # Add disclaimer if verification still didn't pass
        if state.verification_result and not state.verification_result.passed:
            if state.final_answer:
                state.final_answer += "\n\n*Note: Some aspects of this response may require additional verification.*"
        
        return state
    
    async def _simple_synthesis(self, state: AgentState) -> AgentState:
        """Simple synthesis without the full synthesis agent."""
        if not state.retrieved_context:
            state.final_answer = await self._generate_llm_response(
                response_type="no_documents",
                user_message=state.query,
            )
            return state
        
        # Format context
        context_parts = []
        for hit in state.retrieved_context[:5]:  # Limit to top 5
            source = hit.metadata.get("source", hit.id)
            context_parts.append(f"[{source}]:\n{hit.content}")
        
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context_text}

Question: {state.query}

Instructions:
- Answer based ONLY on the provided context
- Cite sources using [source_name] format
- If information is not in the context, say so
- Be concise and direct

Answer:"""
        
        try:
            answer = await self._call_llm(prompt)
            state.final_answer = answer
        except Exception as e:
            logger.error(f"[{self.name}] Synthesis failed: {e}")
            state.final_answer = "I apologize, but I couldn't generate an answer at this time."
        
        return state


def get_orchestrator_agent(
    lm_client: Optional[LMStudioClient] = None,
    include_all_agents: bool = True,
) -> OrchestratorAgent:
    """
    Factory function to create an orchestrator with all sub-agents.
    
    Args:
        lm_client: LLM client (will create default if not provided)
        include_all_agents: If True, instantiate all sub-agents
        
    Returns:
        Configured OrchestratorAgent
    """
    if lm_client is None:
        from app.clients.lmstudio import get_lmstudio_client
        lm_client = get_lmstudio_client()
    
    retrieval_agent = None
    query_decomposer = None
    synthesis_agent = None
    verification_agent = None
    tool_agent = None
    
    if include_all_agents:
        try:
            from app.agents.retrieval_agent import get_retrieval_agent
            retrieval_agent = get_retrieval_agent(lm_client)
        except ImportError:
            logger.warning("Retrieval agent not available")
        
        try:
            from app.agents.query_decomposer import get_query_decomposer
            query_decomposer = get_query_decomposer(lm_client)
        except ImportError:
            logger.warning("Query decomposer not available")
        
        try:
            from app.agents.synthesis_agent import get_synthesis_agent
            synthesis_agent = get_synthesis_agent(lm_client)
        except ImportError:
            logger.warning("Synthesis agent not available")
        
        try:
            from app.agents.verification_agent import get_verification_agent
            verification_agent = get_verification_agent(lm_client)
        except ImportError:
            logger.warning("Verification agent not available")
        
        try:
            from app.agents.tool_agent import get_tool_agent
            tool_agent = get_tool_agent(lm_client)
        except ImportError:
            logger.warning("Tool agent not available")
    
    return OrchestratorAgent(
        lm_client=lm_client,
        retrieval_agent=retrieval_agent,
        query_decomposer=query_decomposer,
        synthesis_agent=synthesis_agent,
        verification_agent=verification_agent,
        tool_agent=tool_agent,
    )
