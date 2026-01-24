"""
Synthesis Agent.

Combines retrieved context, tool results, and sub-query answers
into a coherent final response with proper citations.
"""

import logging
from typing import List, Optional

from app.agents.base import BaseAgent
from app.agents.state import AgentAction, AgentState, ActionType
from app.clients.lmstudio import LMStudioClient
from app.config import settings
from app.models.schemas import RetrievalHit
from app.services.natural_responses import (
    format_tool_results_response,
    get_conversational_response,
)
from app.services.response_generator import generate_response

logger = logging.getLogger(__name__)


SYNTHESIS_PROMPT = """You are an AI assistant that answers questions based on provided context.

CONTEXT:
{context}

{tool_results}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. Cite sources using [source_name] format
3. If information is not in the context, say "I don't have information about that in the provided documents"
4. Be concise and direct
5. If multiple sources support the same point, cite all of them
6. For numerical data, always cite the source

{additional_instructions}

ANSWER:"""


MULTI_QUERY_SYNTHESIS_PROMPT = """You are an AI assistant that synthesizes information from multiple sources.

ORIGINAL QUESTION: {original_question}

SUB-QUESTION RESULTS:
{sub_results}

ADDITIONAL CONTEXT:
{context}

{tool_results}

INSTRUCTIONS:
1. Combine the sub-question answers into a coherent response
2. Maintain citations from the original answers
3. Ensure the final answer fully addresses the original question
4. If sub-questions revealed contradictions, note them
5. Be concise but complete

SYNTHESIZED ANSWER:"""


class SynthesisAgent(BaseAgent):
    """
    Combines context and results into a final answer.
    
    Handles:
    - Single query synthesis from context
    - Multi-query synthesis combining sub-query results
    - Tool result integration
    - Citation enforcement
    """
    
    name: str = "synthesis_agent"
    description: str = "Combines context into final answer"
    
    def __init__(
        self,
        lm_client: LMStudioClient,
        enforce_citations: bool = True,
        max_context_tokens: int = 1000,
        verbose: bool = False,
    ) -> None:
        super().__init__(lm_client, max_iterations=1, verbose=verbose)
        self.enforce_citations = enforce_citations
        self.max_context_tokens = max_context_tokens
        
        # Lazy load compressor
        self._compressor = None
    
    @property
    def compressor(self):
        """Lazy load context compressor."""
        if self._compressor is None:
            try:
                from app.services.context_compressor import get_context_compressor_with_llm
                self._compressor = get_context_compressor_with_llm(self.lm_client)
            except ImportError:
                logger.warning("Context compressor not available")
        return self._compressor
    
    async def run(self, state: AgentState) -> AgentState:
        """
        Generate the final answer from accumulated context.
        """
        logger.debug(f"[{self.name}] Starting synthesis")
        
        # Check if we have sub-query answers to synthesize
        has_sub_answers = any(
            sq.answer is not None 
            for sq in state.sub_queries
        )
        
        if has_sub_answers:
            answer = await self._synthesize_multi_query(state)
        else:
            answer = await self._synthesize_single(state)
        
        state.final_answer = answer
        state.intermediate_answers.append(answer)
        state.add_observation(f"Generated answer ({len(answer)} chars)")
        
        logger.info(f"[{self.name}] Synthesis complete")
        
        return state
    
    async def think(self, state: AgentState) -> str:
        """Analyze what synthesis approach to use."""
        if not state.retrieved_context and not state.tool_results:
            return "No context available - will generate disclaimer"
        
        has_sub_answers = any(sq.answer for sq in state.sub_queries)
        
        if has_sub_answers:
            return "Have sub-query answers - will combine them"
        
        return f"Have {len(state.retrieved_context)} documents - will synthesize"
    
    async def act(self, state: AgentState, thought: str) -> AgentAction:
        """Always synthesize."""
        return AgentAction(
            type=ActionType.SYNTHESIZE,
            reasoning="Generating final answer"
        )
    
    async def observe(self, state: AgentState, action: AgentAction) -> AgentState:
        """Record synthesis completion."""
        state.add_observation("Synthesis complete")
        return state
    
    async def _synthesize_single(self, state: AgentState) -> str:
        """
        Synthesize answer from single query context.
        """
        if not state.retrieved_context:
            response_type = "tool_results" if state.tool_results else "no_documents"
            return await generate_response(
                lm_client=self.lm_client,
                response_type=response_type,
                user_message=state.query,
                tool_results=state.tool_results if state.tool_results else None,
                allow_fallback_templates=False,
            )
        
        # Compress context if available
        context_text = await self._prepare_context(state.retrieved_context, state.query)
        
        # Format tool results if any
        tool_text = self._format_tool_results(state.tool_results)
        
        # Build additional instructions
        additional = self._get_additional_instructions()
        
        prompt = SYNTHESIS_PROMPT.format(
            context=context_text,
            tool_results=tool_text,
            question=state.query,
            additional_instructions=additional
        )
        
        try:
            answer = await self._call_llm(prompt)
            return answer.strip()
        except Exception as e:
            logger.error(f"[{self.name}] Synthesis failed: {e}")
            return "I apologize, but I couldn't generate an answer at this time."
    
    async def _synthesize_multi_query(self, state: AgentState) -> str:
        """
        Synthesize answer from multiple sub-query results.
        """
        # Collect sub-query answers
        sub_results = []
        for i, sq in enumerate(state.sub_queries):
            if sq.answer:
                sub_results.append(f"Q{i+1}: {sq.query}\nA{i+1}: {sq.answer}")
            elif sq.results:
                # Generate answer for sub-query if not already done
                sub_answer = await self._generate_sub_answer(sq, state)
                sq.answer = sub_answer
                sub_results.append(f"Q{i+1}: {sq.query}\nA{i+1}: {sub_answer}")
        
        sub_results_text = "\n\n".join(sub_results)
        
        # Prepare additional context
        context_text = await self._prepare_context(state.retrieved_context, state.query)
        tool_text = self._format_tool_results(state.tool_results)
        
        prompt = MULTI_QUERY_SYNTHESIS_PROMPT.format(
            original_question=state.original_query or state.query,
            sub_results=sub_results_text,
            context=context_text,
            tool_results=tool_text
        )
        
        try:
            answer = await self._call_llm(prompt)
            return answer.strip()
        except Exception as e:
            logger.error(f"[{self.name}] Multi-query synthesis failed: {e}")
            # Fall back to single synthesis
            return await self._synthesize_single(state)
    
    async def _generate_sub_answer(self, sub_query, state: AgentState) -> str:
        """Generate answer for a single sub-query."""
        if not sub_query.results:
            return "No information found."
        
        context_text = await self._prepare_context(sub_query.results, sub_query.query)
        
        prompt = f"""Based on this context, briefly answer the question.

Context:
{context_text}

Question: {sub_query.query}

Brief answer (1-2 sentences, cite sources):"""
        
        try:
            return await self._call_llm(prompt)
        except Exception:
            return "Unable to generate answer."
    
    async def _prepare_context(
        self, 
        hits: List[RetrievalHit], 
        query: str
    ) -> str:
        """
        Prepare context text from retrieval hits.
        
        Optionally compresses context to fit token budget.
        """
        if not hits:
            return "No relevant documents found."
        
        # Try to compress if compressor available
        if self.compressor:
            try:
                compressed = await self.compressor.compress(
                    hits=hits,
                    query=query,
                    use_llm_refinement=False,  # Skip LLM for speed
                    max_facts=10
                )
                
                if compressed:
                    parts = []
                    for fact in compressed:
                        source = fact.source_name or fact.source_id
                        parts.append(f"[{source}]: {fact.text}")
                    return "\n\n".join(parts)
            except Exception as e:
                logger.debug(f"Compression failed, using raw context: {e}")
        
        # Fall back to raw context
        parts = []
        for hit in hits[:7]:  # Limit documents
            source = hit.metadata.get("source", hit.id)
            content = hit.content
            
            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."
            
            parts.append(f"[{source}]:\n{content}")
        
        return "\n\n".join(parts)
    
    def _format_tool_results(self, tool_results: dict) -> str:
        """Format tool results for inclusion in prompt."""
        if not tool_results:
            return ""
        
        parts = ["TOOL RESULTS:"]
        for tool_name, result in tool_results.items():
            parts.append(f"- {tool_name}: {result}")
        
        return "\n".join(parts)
    
    def _get_additional_instructions(self) -> str:
        """Get additional instructions based on settings."""
        instructions = []
        
        if self.enforce_citations and settings.rag.citation_required:
            instructions.append(
                "IMPORTANT: Every factual claim must be cited with [source_name]"
            )
        
        if settings.rag.min_confidence_threshold > 0:
            instructions.append(
                "If you're uncertain about any information, express that uncertainty"
            )
        
        return "\n".join(instructions)
    
    def _generate_no_context_response(self, query: str, state: Optional[AgentState] = None) -> str:
        """Generate response when no context is available."""
        # Check if this should have been gated earlier (chitchat)
        if state and state.intent == "chitchat":
            return get_conversational_response(query)
        
        # Check if tools were used but no documents found
        if state and state.tool_results:
            # Tools were executed - respond based on tool results
            return format_tool_results_response(state.tool_results)
        
        # Actual no-context situation - provide helpful guidance
        return (
            "I couldn't find relevant information in the documents for your question. "
            "This could mean:\n"
            "- The documents don't contain information about this topic\n"
            "- You might want to rephrase your question\n"
            "- The relevant documents may not have been uploaded yet\n\n"
            "Would you like to try a different question, or can I help with something else?"
        )


def get_synthesis_agent(
    lm_client: Optional[LMStudioClient] = None,
) -> SynthesisAgent:
    """Factory function to create synthesis agent."""
    if lm_client is None:
        from app.clients.lmstudio import get_lmstudio_client
        lm_client = get_lmstudio_client()
    
    return SynthesisAgent(
        lm_client=lm_client,
        enforce_citations=settings.rag.citation_required,
    )
