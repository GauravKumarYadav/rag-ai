"""
Chat Service - Optimized for Small Model RAG.

This service implements the full optimized RAG pipeline:
1. State management (structured state instead of raw history)
2. Intent classification and query rewriting
3. Hybrid search (BM25 + Vector) with reranking + MMR
4. Knowledge graph expansion for entity disambiguation
5. Context compression to dense bullets
6. Evidence validation and guardrails
7. Citation enforcement and verification
8. Token budgeting
9. Sliding window memory with episodic extraction
10. Answer verification for hallucination detection
"""

import logging
import re
from typing import AsyncGenerator, List, Optional, Tuple

from fastapi import BackgroundTasks

from app.clients.lmstudio import LMStudioClient, get_lmstudio_client
from app.config import settings
from app.memory.conversation_state import (
    ConversationState,
    ConversationStateManager,
    get_state_manager,
)
from app.memory.long_term import LongTermMemory, get_long_term_memory
from app.memory.session_buffer import SessionBuffer, get_session_buffer
from app.models.client import ClientStore, get_client_store
from app.models.schemas import ChatRequest, RetrievalHit
from app.rag.retriever import Retriever, get_retriever
from app.rag.vector_store import get_client_vector_store
from app.services.client_extractor import ClientExtractor, get_client_extractor
from app.services.context_compressor import (
    CompressedFact,
    ContextCompressor,
    get_context_compressor_with_llm,
)
from app.services.evidence_validator import (
    EvidenceAssessment,
    EvidenceValidator,
    get_evidence_validator_with_llm,
)
from app.services.prompt_builder import build_messages, build_optimized_messages
from app.services.query_processor import (
    Intent,
    QueryProcessor,
    QueryResult,
    get_query_processor,
)
from app.services.token_budgeter import TokenBudgeter, get_token_budgeter

# New components for enhanced RAG
from app.rag.hybrid_search import HybridSearch, get_hybrid_search
from app.services.citation_extractor import CitationExtractor, get_citation_extractor
from app.services.answer_verifier import AnswerVerifier, get_answer_verifier

logger = logging.getLogger(__name__)


# Legacy patterns kept for backward compatibility
CHITCHAT_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings|yo)[\s!?.]*$",
    r"^(how are you|how's it going|what's up|sup)[\s!?.]*$",
    r"^(good morning|good afternoon|good evening|good night)[\s!?.]*$",
    r"^(thanks|thank you|thx|ty)[\s!?.]*$",
    r"^(bye|goodbye|see you|later|cya)[\s!?.]*$",
    r"^(yes|no|ok|okay|sure|alright|fine|great|cool)[\s!?.]*$",
    r"^(help|what can you do|who are you)[\s!?.]*$",
]

CONTEXT_KEYWORDS = [
    "document", "file", "client", "show", "tell me about", "what is",
    "find", "search", "look up", "information", "details", "data",
    "analyze", "summary", "summarize", "explain", "list", "get",
    "aadhar", "aadhaar", "card", "pdf", "report", "record",
]


def _needs_rag_retrieval(message: str) -> bool:
    """
    Legacy function for backward compatibility.
    Use QueryProcessor for full intent classification.
    """
    msg_lower = message.lower().strip()
    
    for pattern in CHITCHAT_PATTERNS:
        if re.match(pattern, msg_lower, re.IGNORECASE):
            return False
    
    for keyword in CONTEXT_KEYWORDS:
        if keyword in msg_lower:
            return True
    
    if len(msg_lower) < 20 and "?" not in msg_lower:
        return False
    
    return True


class ChatService:
    """
    Chat service with small model RAG optimization.
    
    Enhanced optimized pipeline:
    1. Load/build state block from conversation state
    2. Classify intent and rewrite query
    3. Hybrid search (BM25 + Vector) with reranking + MMR
    4. Knowledge graph expansion (if enabled)
    5. Compress context to dense bullet facts
    6. Validate evidence and add guardrails
    7. Fit to token budget
    8. Build optimized prompt with citation enforcement
    9. Generate response
    10. Verify answer against context
    11. Update state and memory
    
    Falls back to legacy pipeline if optimization disabled.
    """
    
    def __init__(
        self,
        lm_client: LMStudioClient,
        retriever: Retriever,
        session_buffer: SessionBuffer,
        long_term: LongTermMemory,
        client_extractor: Optional[ClientExtractor] = None,
        client_store: Optional[ClientStore] = None,
        state_manager: Optional[ConversationStateManager] = None,
        query_processor: Optional[QueryProcessor] = None,
        context_compressor: Optional[ContextCompressor] = None,
        evidence_validator: Optional[EvidenceValidator] = None,
        token_budgeter: Optional[TokenBudgeter] = None,
        citation_extractor: Optional[CitationExtractor] = None,
        answer_verifier: Optional[AnswerVerifier] = None,
        use_optimized_pipeline: bool = True,
    ) -> None:
        self.lm_client = lm_client
        self.retriever = retriever
        self.session_buffer = session_buffer
        self.long_term = long_term
        self.client_extractor = client_extractor or get_client_extractor()
        self.client_store = client_store or get_client_store()
        
        # Small model optimization components
        self.state_manager = state_manager or get_state_manager()
        self.query_processor = query_processor or get_query_processor()
        self.context_compressor = context_compressor or get_context_compressor_with_llm(lm_client)
        self.evidence_validator = evidence_validator or get_evidence_validator_with_llm(lm_client)
        self.token_budgeter = token_budgeter or get_token_budgeter()
        
        # New components for enhanced RAG quality
        self.citation_extractor = citation_extractor or get_citation_extractor()
        self.answer_verifier = answer_verifier or get_answer_verifier(lm_client)
        
        self.use_optimized_pipeline = use_optimized_pipeline

    async def _build_system_context(self) -> str:
        """Build context about the system's clients and documents for the LLM."""
        clients = await self.client_store.list_all()
        
        if not clients:
            return "\n\n[System Info: No clients registered. No documents available.]"
        
        context_parts = ["\n\n[System Information:"]
        context_parts.append(f"Total clients: {len(clients)}")
        
        for client in clients:
            client_store = get_client_vector_store(client.id)
            chunk_count = client_store.docs.count()
            memory_count = client_store.memories.count()
            
            doc_sources = []
            if chunk_count > 0:
                try:
                    all_docs = client_store.docs.get(include=["metadatas"])
                    sources = set()
                    for meta in all_docs.get("metadatas", []):
                        if meta and meta.get("source"):
                            sources.add(meta["source"])
                    doc_sources = list(sources)
                except Exception:
                    pass
            
            doc_count = len(doc_sources) if doc_sources else 0
            
            context_parts.append(f"\nClient: {client.name} (ID: {client.id})")
            context_parts.append(f"  - Documents: {doc_count} files ({chunk_count} chunks)")
            if doc_sources:
                context_parts.append(f"  - Document files: {', '.join(doc_sources)}")
            context_parts.append(f"  - Memories: {memory_count}")
            if client.aliases:
                context_parts.append(f"  - Aliases: {', '.join(client.aliases)}")
        
        context_parts.append("]")
        return "\n".join(context_parts)

    async def _detect_client(self, message: str, conversation_id: str) -> Optional[str]:
        """Detect client from message using LLM and match against known clients."""
        extraction = await self.client_extractor.extract_client(message)
        
        if not extraction or extraction.get("confidence", 0) < 0.5:
            return None
        
        client_name = extraction.get("client_name")
        client_id = extraction.get("client_id")
        
        if client_id:
            client = await self.client_store.get(client_id)
            if client:
                return client.id
        
        if client_name:
            matches = await self.client_store.search(client_name)
            if matches:
                return matches[0].id
        
        return None

    async def handle_chat(
        self, 
        request: ChatRequest, 
        background_tasks: BackgroundTasks | None = None,
        allowed_clients: Optional[set] = None,
    ) -> Tuple[str | AsyncGenerator[str, None], List[RetrievalHit]]:
        """
        Handle chat request with optimized or legacy pipeline.
        
        Args:
            request: The chat request
            background_tasks: Optional background tasks for async operations
            allowed_clients: Set of client IDs the user can access (for authorization)
        """
        if self.use_optimized_pipeline:
            return await self._handle_chat_optimized(request, background_tasks, allowed_clients)
        else:
            return await self._handle_chat_legacy(request, background_tasks, allowed_clients)

    async def _handle_chat_optimized(
        self, 
        request: ChatRequest, 
        background_tasks: BackgroundTasks | None = None,
        allowed_clients: Optional[set] = None,
    ) -> Tuple[str | AsyncGenerator[str, None], List[RetrievalHit]]:
        """
        Enhanced optimized chat pipeline for small models.
        
        Steps:
        1. Load conversation state
        2. Process query (intent + rewrite)
        3. Hybrid search (BM25 + Vector) with reranking + MMR
        4. Knowledge graph expansion (optional)
        5. Compress to bullets
        6. Validate evidence
        7. Fit to token budget
        8. Build optimized prompt with citation enforcement
        9. Generate response
        10. Verify answer against context
        11. Update state and memory
        """
        from app.core.metrics import (
            record_cross_client_filter, 
            record_stage_executed,
            record_stage_skipped,
            record_intent,
        )
        
        conversation_id = request.conversation_id
        
        # Step 1: Load conversation state
        state = await self.state_manager.get_state(conversation_id)
        state_block = self.state_manager.build_state_block(state)
        logger.debug(f"[Optimized] State block: {state_block[:100] if state_block else 'empty'}...")
        
        # Step 2: Process query (intent classification + rewriting)
        query_result = await self.query_processor.process(request.message, state)
        record_intent(query_result.intent.value)
        logger.debug(f"[Optimized] Intent: {query_result.intent.value}, needs_retrieval: {query_result.needs_retrieval}")
        
        # Initialize for non-retrieval path
        compressed_facts: List[CompressedFact] = []
        retrieved: List[RetrievalHit] = []
        evidence_assessment: Optional[EvidenceAssessment] = None
        evidence_disclaimer: Optional[str] = None
        
        # Use the client_id from request (already validated by route)
        # Default to 'global' if not specified
        client_id = request.client_id or "global"
        
        # =================================================================
        # INTENT-BASED GATING: Skip retrieval for chitchat
        # =================================================================
        if query_result.intent == Intent.CHITCHAT:
            record_stage_skipped("retrieval", "chitchat")
            logger.info(f"[Gating] Skipping retrieval - chitchat intent detected")
            
            # Generate response without retrieval
            return await self._generate_without_retrieval(
                request=request,
                state=state,
                background_tasks=background_tasks,
            )
        
        # =================================================================
        # FOLLOW-UP GATING: Only retrieve if references exist
        # =================================================================
        if query_result.intent == Intent.FOLLOW_UP and not query_result.resolved_references:
            record_stage_skipped("retrieval", "no_refs")
            logger.info(f"[Gating] Skipping retrieval - follow-up without resolvable references")
            
            # Generate from state only
            return await self._generate_from_state(
                request=request,
                state=state,
                background_tasks=background_tasks,
            )
        
        # Step 3-7: Retrieval pipeline (only if needed)
        if query_result.needs_retrieval and request.top_k > 0:
            record_stage_executed("retrieval")
            
            # Record that we're applying client filter
            record_cross_client_filter(client_id)
            
            # Update state with client context
            if client_id and client_id != "global":
                client = await self.client_store.get(client_id)
                if client:
                    await self.state_manager.set_client_context(conversation_id, client.name)
            
            # Step 3: Hybrid Search (BM25 + Vector) with reranking + MMR
            # ALWAYS filter by client_id - this is the security boundary
            search_query = query_result.search_query
            
            # Use hybrid search if BM25 is enabled
            if settings.rag.bm25_enabled:
                # Client-specific hybrid search with hard client filter
                try:
                    hybrid = get_hybrid_search(client_id=client_id)
                    retrieved = hybrid.search(
                        query=search_query,
                        top_k=settings.rag.rerank_top_k,
                        fetch_k=settings.rag.initial_fetch_k,
                        use_reranker=settings.rag.reranker_enabled,
                    )
                    logger.debug(f"[Optimized] Hybrid search for client {client_id}: {len(retrieved)} docs")
                except Exception as e:
                    logger.warning(f"Hybrid search failed, falling back: {e}")
                    retrieved = self.retriever.search_with_client_filter(
                        query=search_query,
                        client_id=client_id,
                        top_k=settings.rag.rerank_top_k,
                        fetch_k=settings.rag.initial_fetch_k,
                        metadata_filters=request.metadata_filters,
                    )
            else:
                # Vector retrieval with client filter
                retrieved = self.retriever.search_with_client_filter(
                    query=search_query,
                    client_id=client_id,
                    top_k=settings.rag.rerank_top_k,
                    fetch_k=settings.rag.initial_fetch_k,
                    metadata_filters=request.metadata_filters,
                )
                logger.debug(f"[Optimized] Vector retrieval for client {client_id}: {len(retrieved)} docs")
            
            # Step 4: Knowledge Graph Expansion (with gating)
            # Only expand KG when initial retrieval is weak
            if settings.rag.knowledge_graph_enabled and client_id and retrieved:
                should_expand_kg = self._should_expand_kg(retrieved, query_result)
                
                if should_expand_kg:
                    try:
                        from app.knowledge.graph_query import get_graph_query_expander
                        from app.core.metrics import record_kg_expansion
                        
                        record_kg_expansion("low_recall")
                        record_stage_executed("kg_expansion")
                        
                        kg_expander = get_graph_query_expander()
                        _, expansion = kg_expander.enrich_retrieval(search_query, retrieved, client_id)
                        
                        # Log KG expansion info
                        if expansion.identified_entities:
                            logger.debug(
                                f"[Optimized] KG identified {len(expansion.identified_entities)} entities, "
                                f"found {len(expansion.related_entities)} related"
                            )
                    except Exception as e:
                        logger.warning(f"Knowledge graph expansion failed: {e}")
                else:
                    record_stage_skipped("kg_expansion", "good_recall")
                    logger.debug("[Gating] Skipped KG expansion - good initial recall")
            
            # Step 5: Compress to bullets
            if retrieved:
                compressed_facts = await self.context_compressor.compress(
                    hits=retrieved,
                    query=search_query,
                    use_llm_refinement=True,
                    max_facts=15,
                )
                logger.debug(f"[Optimized] Compressed to {len(compressed_facts)} facts")
            
            # Step 6: Validate evidence
            evidence_assessment = self.evidence_validator.assess_evidence(retrieved)
            if evidence_assessment.disclaimer:
                evidence_disclaimer = evidence_assessment.disclaimer
            logger.debug(f"[Optimized] Evidence confidence: {evidence_assessment.confidence:.2f}")
            
            # Check for contradictions
            if compressed_facts:
                contradictions = await self.evidence_validator.check_contradictions(compressed_facts)
                if contradictions:
                    contradiction_warning = self.evidence_validator.format_contradictions_warning(contradictions)
                    if contradiction_warning and evidence_disclaimer:
                        evidence_disclaimer += "\n" + contradiction_warning
                    elif contradiction_warning:
                        evidence_disclaimer = contradiction_warning
            
            # Step 7: Fit to token budget
            if compressed_facts:
                budget_result = self.token_budgeter.fit_to_budget(compressed_facts)
                compressed_facts = budget_result.facts
                logger.debug(f"[Optimized] Budget: {budget_result.total_tokens}/{budget_result.budget} tokens")
        
        # Get sliding window messages and running summary
        recent_messages, running_summary = self.session_buffer.get_with_summary(conversation_id)
        
        # Get episodic memories
        episodic_memories: List[str] = []
        if state.decisions_made:
            episodic_memories.extend(state.decisions_made[-3:])
        
        # Get provider type
        provider = getattr(self.lm_client, "provider", "ollama")
        
        # Step 8: Build optimized prompt with citation enforcement
        enforce_citations = settings.rag.citation_required and compressed_facts
        
        messages = build_optimized_messages(
            user_text=request.message,
            state_block=state_block if state_block else None,
            compressed_facts=compressed_facts if compressed_facts else None,
            recent_messages=recent_messages if recent_messages else None,
            running_summary=running_summary,
            episodic_memories=episodic_memories if episodic_memories else None,
            evidence_disclaimer=evidence_disclaimer,
            images=request.images if request.images else None,
            system_prompt=request.system_prompt,
            provider=provider,
            enforce_citations=enforce_citations,
        )
        
        # Step 9: Generate response
        result = await self.lm_client.chat(messages, stream=request.stream)
        
        if request.stream:
            return self._wrap_stream_optimized(request, result, retrieved, background_tasks), retrieved
        
        text = result
        
        # Step 10: Verify answer against context (for factual queries)
        if settings.rag.verification_enabled and retrieved and query_result.intent == Intent.QUESTION:
            try:
                verification = await self.answer_verifier.verify(text, retrieved)
                
                # Add disclaimer if needed
                if verification.disclaimer:
                    text = text + "\n\n" + verification.disclaimer
                
                # Log verification results
                if verification.unsupported_claims:
                    logger.warning(
                        f"[Optimized] Unsupported claims detected: {verification.unsupported_claims[:2]}"
                    )
                
                # Check citation coverage
                if settings.rag.citation_required:
                    passes, citation_analysis = self.citation_extractor.passes_coverage_threshold(text, retrieved)
                    if not passes:
                        coverage_warning = self.citation_extractor.format_coverage_warning(citation_analysis)
                        if coverage_warning:
                            text = text + "\n\n" + coverage_warning
                            logger.debug(f"[Optimized] Citation coverage: {citation_analysis.coverage_ratio:.1%}")
                
            except Exception as e:
                logger.error(f"Answer verification failed: {e}")
        
        # Step 11: Update state and memory
        await self._post_turn_optimized(request, text, background_tasks)
        return text, retrieved

    async def _handle_chat_legacy(
        self, 
        request: ChatRequest, 
        background_tasks: BackgroundTasks | None = None,
        allowed_clients: Optional[set] = None,
    ) -> Tuple[str | AsyncGenerator[str, None], List[RetrievalHit]]:
        """
        Legacy chat pipeline (backward compatible).
        
        Now enforces client_id filtering for security.
        """
        from app.core.metrics import record_cross_client_filter, record_stage_executed
        
        history = self.session_buffer.get(request.conversation_id)
        
        needs_rag = _needs_rag_retrieval(request.message)
        logger.debug(f"[Legacy] Message needs RAG: {needs_rag}")
        
        # Use client_id from request (already validated by route)
        # Default to 'global' if not specified
        client_id = request.client_id or "global"
        
        retrieved = []
        memory_hits = []
        
        if needs_rag and request.top_k > 0:
            record_stage_executed("retrieval")
            record_cross_client_filter(client_id)
            
            # ALWAYS search with client filter - security boundary
            client_store = get_client_vector_store(client_id)
            retrieved = client_store.query(
                query=request.message, 
                top_k=request.top_k, 
                collection="documents"
            )
            logger.debug(f"[Legacy] Retrieved {len(retrieved)} docs for client {client_id}")
            
            memory_hits = self.long_term.retrieve(query=request.message)
        
        system_context = ""
        client_context = ""
        if needs_rag:
            system_context = await self._build_system_context()
            if client_id:
                client = await self.client_store.get(client_id)
                if client:
                    client_context = f"\n\n[Current client context: {client.name}]"
        
        system_prompt = (request.system_prompt or "") + system_context + client_context
        provider = getattr(self.lm_client, "provider", "ollama")
        
        messages = build_messages(
            system_prompt=system_prompt,
            user_text=request.message,
            images=request.images,
            retrieved_docs=retrieved,
            memory_hits=memory_hits,
            session_messages=history,
            provider=provider,
        )

        result = await self.lm_client.chat(messages, stream=request.stream)
        if request.stream:
            return self._wrap_stream(request, result, retrieved, background_tasks), retrieved

        text = result
        await self._post_turn(request, text, background_tasks)
        return text, retrieved

    async def _post_turn(
        self,
        request: ChatRequest,
        assistant_text: str,
        background_tasks: BackgroundTasks | None,
    ) -> None:
        """Legacy post-turn processing."""
        self.session_buffer.add(request.conversation_id, "user", request.message)
        self.session_buffer.add(request.conversation_id, "assistant", assistant_text)

        if self._should_summarize(request.conversation_id):
            if background_tasks:
                background_tasks.add_task(
                    self.long_term.summarize_and_store,
                    request.conversation_id,
                    self.session_buffer.get(request.conversation_id),
                )
            else:
                await self.long_term.summarize_and_store(
                    request.conversation_id, self.session_buffer.get(request.conversation_id)
                )

    async def _post_turn_optimized(
        self,
        request: ChatRequest,
        assistant_text: str,
        background_tasks: BackgroundTasks | None,
    ) -> None:
        """
        Optimized post-turn processing with state updates.
        """
        conversation_id = request.conversation_id
        
        # Add to session buffer
        self.session_buffer.add(conversation_id, "user", request.message)
        self.session_buffer.add(conversation_id, "assistant", assistant_text)
        
        # Update conversation state
        await self.state_manager.update_state(
            conversation_id=conversation_id,
            user_message=request.message,
            assistant_response=assistant_text,
        )
        
        # Check if we need to generate running summary
        if self.session_buffer.needs_summarization(conversation_id):
            older_messages = self.session_buffer.get_older(conversation_id)
            if older_messages:
                if background_tasks:
                    background_tasks.add_task(
                        self._generate_running_summary,
                        conversation_id,
                        older_messages,
                    )
                else:
                    await self._generate_running_summary(conversation_id, older_messages)
        
        # Extract and store episodic memories periodically
        if settings.session.episodic_memory_enabled:
            history = self.session_buffer.get(conversation_id)
            if len(history) % 6 == 0:  # Every 3 turns
                if background_tasks:
                    background_tasks.add_task(
                        self.long_term.extract_and_store_episodics,
                        conversation_id,
                        history[-6:],  # Last 3 turns
                    )
                else:
                    memories = await self.long_term.extract_and_store_episodics(
                        conversation_id, history[-6:]
                    )
                    # Update state with extracted decisions
                    for mem in memories:
                        if mem.memory_type == "decision":
                            await self.state_manager.add_decision(conversation_id, mem.content)

    async def _generate_running_summary(
        self,
        conversation_id: str,
        messages: list,
    ) -> None:
        """Generate and store running summary for older messages."""
        try:
            summary = await self.long_term.generate_running_summary(messages)
            if summary:
                self.session_buffer.set_running_summary(conversation_id, summary)
                await self.state_manager.set_running_summary(conversation_id, summary)
                logger.debug(f"Generated running summary for {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to generate running summary: {e}")
    
    async def _generate_without_retrieval(
        self,
        request: ChatRequest,
        state: ConversationState,
        background_tasks: BackgroundTasks | None = None,
    ) -> Tuple[str | AsyncGenerator[str, None], List[RetrievalHit]]:
        """
        Generate response without RAG retrieval.
        
        Used for chitchat and queries that don't need document context.
        """
        logger.debug(f"[Gating] Generating response without retrieval")
        
        history = self.session_buffer.get(request.conversation_id)
        
        # Build simple conversational prompt
        system_prompt = request.system_prompt or ""
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant. Be concise and friendly. "
                "If you don't have specific information about documents, "
                "be honest about what you don't know."
            )
        
        provider = getattr(self.lm_client, "provider", "ollama")
        
        messages = build_messages(
            system_prompt=system_prompt,
            user_text=request.message,
            images=request.images,
            retrieved_docs=[],  # No retrieved docs
            memory_hits=[],
            session_messages=history[-6:],  # Limited history
            provider=provider,
        )
        
        result = await self.lm_client.chat(messages, stream=request.stream)
        
        if request.stream:
            return self._wrap_stream_optimized(request, result, [], background_tasks), []
        
        await self._post_turn_optimized(request, result, background_tasks)
        return result, []
    
    async def _generate_from_state(
        self,
        request: ChatRequest,
        state: ConversationState,
        background_tasks: BackgroundTasks | None = None,
    ) -> Tuple[str | AsyncGenerator[str, None], List[RetrievalHit]]:
        """
        Generate response using conversation state only (no new retrieval).
        
        Used for follow-ups that reference previous context already in state.
        """
        logger.debug(f"[Gating] Generating response from state only")
        
        history = self.session_buffer.get(request.conversation_id)
        state_block = self.state_manager.build_state_block(state)
        
        # Build prompt with state context but no new retrieval
        system_prompt = request.system_prompt or ""
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant. Answer based on the conversation context "
                "provided. If you need to refer to documents, use information already "
                "discussed in the conversation."
            )
        
        # Add state context to system prompt
        if state_block:
            system_prompt = f"{system_prompt}\n\nConversation context:\n{state_block}"
        
        provider = getattr(self.lm_client, "provider", "ollama")
        
        messages = build_messages(
            system_prompt=system_prompt,
            user_text=request.message,
            images=request.images,
            retrieved_docs=[],  # No new retrieval
            memory_hits=[],
            session_messages=history,
            provider=provider,
        )
        
        result = await self.lm_client.chat(messages, stream=request.stream)
        
        if request.stream:
            return self._wrap_stream_optimized(request, result, [], background_tasks), []
        
        await self._post_turn_optimized(request, result, background_tasks)
        return result, []
    
    def _should_expand_kg(
        self, 
        retrieved: List[RetrievalHit], 
        query_result: QueryResult,
    ) -> bool:
        """
        Determine if Knowledge Graph expansion should be triggered.
        
        Expand KG when:
        - Too few results (< 2)
        - Low confidence scores (top result score > 0.5 for distance metrics)
        - Query intent is exploratory
        
        Don't expand when:
        - Good recall (3+ high-confidence results)
        - Follow-up query (usually doesn't need expansion)
        - High confidence retrieval
        
        Args:
            retrieved: Retrieved hits from initial search
            query_result: Query processing result with intent
            
        Returns:
            True if KG expansion should be triggered
        """
        # Too few results - definitely expand
        if len(retrieved) < 2:
            return True
        
        # Follow-ups usually don't need KG expansion
        if query_result.intent == Intent.FOLLOW_UP:
            return False
        
        # Check confidence of top results
        if retrieved:
            top_score = retrieved[0].score
            
            # If using distance metrics (lower is better)
            # Expand if top result has low confidence (high distance)
            if top_score > 0.5:
                return True
            
            # If we have 3+ results with good scores, skip expansion
            if len(retrieved) >= 3:
                third_score = retrieved[2].score
                if third_score < 0.4:  # Good confidence for top 3
                    return False
        
        # Default: don't expand (save resources)
        return False

    def _should_summarize(self, conversation_id: str) -> bool:
        """Check if conversation needs summarization."""
        history = self.session_buffer.get(conversation_id)
        return len(history) >= self.session_buffer.max_messages

    def _wrap_stream(
        self,
        request: ChatRequest,
        stream: AsyncGenerator[str, None],
        retrieved: List[RetrievalHit],
        background_tasks: BackgroundTasks | None,
    ) -> AsyncGenerator[str, None]:
        """Wrap stream for legacy pipeline."""
        async def generator() -> AsyncGenerator[str, None]:
            buffer = ""
            async for chunk in stream:
                buffer += chunk
                encoded_chunk = chunk.replace('\n', '\\n')
                yield f"data: {encoded_chunk}\n\n"
            await self._post_turn(request, buffer, background_tasks)

        return generator()

    def _wrap_stream_optimized(
        self,
        request: ChatRequest,
        stream: AsyncGenerator[str, None],
        retrieved: List[RetrievalHit],
        background_tasks: BackgroundTasks | None,
    ) -> AsyncGenerator[str, None]:
        """Wrap stream for optimized pipeline."""
        async def generator() -> AsyncGenerator[str, None]:
            buffer = ""
            async for chunk in stream:
                buffer += chunk
                encoded_chunk = chunk.replace('\n', '\\n')
                yield f"data: {encoded_chunk}\n\n"
            await self._post_turn_optimized(request, buffer, background_tasks)

        return generator()


def get_chat_service(use_optimized: bool = True) -> ChatService:
    """
    Get chat service instance.
    
    Args:
        use_optimized: If True, uses the optimized small model pipeline.
                      If False, uses the legacy pipeline.
    """
    lm_client = get_lmstudio_client()
    
    return ChatService(
        lm_client=lm_client,
        retriever=get_retriever(),
        session_buffer=get_session_buffer(),
        long_term=get_long_term_memory(),
        client_extractor=get_client_extractor(),
        client_store=get_client_store(),
        state_manager=get_state_manager(),
        query_processor=get_query_processor(),
        context_compressor=get_context_compressor_with_llm(lm_client),
        evidence_validator=get_evidence_validator_with_llm(lm_client),
        token_budgeter=get_token_budgeter(),
        citation_extractor=get_citation_extractor(),
        answer_verifier=get_answer_verifier(lm_client),
        use_optimized_pipeline=use_optimized,
    )
