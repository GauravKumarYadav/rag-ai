"""
Chat Service - Optimized for Small Model RAG.

This service implements the full optimized RAG pipeline:
1. State management (structured state instead of raw history)
2. Intent classification and query rewriting
3. Reranking with MMR diversity
4. Context compression to dense bullets
5. Evidence validation and guardrails
6. Token budgeting
7. Sliding window memory with episodic extraction
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
    
    New optimized pipeline:
    1. Load/build state block from conversation state
    2. Classify intent and rewrite query
    3. Retrieve with reranking + MMR diversity
    4. Compress context to dense bullet facts
    5. Validate evidence and add guardrails
    6. Fit to token budget
    7. Build optimized prompt
    8. Generate response
    9. Update state and memory
    
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
        self, request: ChatRequest, background_tasks: BackgroundTasks | None = None
    ) -> Tuple[str | AsyncGenerator[str, None], List[RetrievalHit]]:
        """
        Handle chat request with optimized or legacy pipeline.
        """
        if self.use_optimized_pipeline:
            return await self._handle_chat_optimized(request, background_tasks)
        else:
            return await self._handle_chat_legacy(request, background_tasks)

    async def _handle_chat_optimized(
        self, request: ChatRequest, background_tasks: BackgroundTasks | None = None
    ) -> Tuple[str | AsyncGenerator[str, None], List[RetrievalHit]]:
        """
        Optimized chat pipeline for small models.
        
        Steps:
        1. Load conversation state
        2. Process query (intent + rewrite)
        3. Retrieve with reranking + MMR
        4. Compress to bullets
        5. Validate evidence
        6. Fit to token budget
        7. Build optimized prompt
        8. Generate response
        9. Update state
        """
        conversation_id = request.conversation_id
        
        # Step 1: Load conversation state
        state = await self.state_manager.get_state(conversation_id)
        state_block = self.state_manager.build_state_block(state)
        logger.debug(f"[Optimized] State block: {state_block[:100] if state_block else 'empty'}...")
        
        # Step 2: Process query (intent classification + rewriting)
        query_result = await self.query_processor.process(request.message, state)
        logger.debug(f"[Optimized] Intent: {query_result.intent.value}, needs_retrieval: {query_result.needs_retrieval}")
        
        # Initialize for non-retrieval path
        compressed_facts: List[CompressedFact] = []
        retrieved: List[RetrievalHit] = []
        evidence_assessment: Optional[EvidenceAssessment] = None
        evidence_disclaimer: Optional[str] = None
        
        # Step 3-6: Retrieval pipeline (only if needed)
        if query_result.needs_retrieval and request.top_k > 0:
            # Detect client context
            client_id = request.client_id
            if not client_id:
                client_id = await self._detect_client(request.message, conversation_id)
            
            # Update state with client context
            if client_id:
                client = await self.client_store.get(client_id)
                if client:
                    await self.state_manager.set_client_context(conversation_id, client.name)
            
            # Step 3: Retrieve with reranking + MMR
            search_query = query_result.search_query
            
            # Global retrieval with optimized pipeline
            retrieved = self.retriever.search_with_mmr(
                query=search_query,
                top_k=settings.rag.rerank_top_k,
                fetch_k=settings.rag.initial_fetch_k,
                metadata_filters=request.metadata_filters,
            )
            logger.debug(f"[Optimized] Global retrieval: {len(retrieved)} docs")
            
            # Client-specific retrieval
            if client_id:
                client_store = get_client_vector_store(client_id)
                if client_store.docs.count() > 0:
                    client_hits = client_store.query(
                        query=search_query,
                        top_k=settings.rag.rerank_top_k,
                        collection="documents"
                    )
                    # Prepend client-specific results
                    retrieved = client_hits + retrieved
                    logger.debug(f"[Optimized] With client docs: {len(retrieved)} total")
            else:
                # Search all clients
                all_clients = await self.client_store.list_all()
                for client in all_clients:
                    client_store = get_client_vector_store(client.id)
                    if client_store.docs.count() > 0:
                        hits = client_store.query(
                            query=search_query,
                            top_k=settings.rag.rerank_top_k,
                            collection="documents"
                        )
                        for hit in hits:
                            hit.metadata["client_name"] = client.name
                        retrieved.extend(hits)
            
            # Step 4: Compress to bullets
            if retrieved:
                compressed_facts = await self.context_compressor.compress(
                    hits=retrieved,
                    query=search_query,
                    use_llm_refinement=True,
                    max_facts=15,
                )
                logger.debug(f"[Optimized] Compressed to {len(compressed_facts)} facts")
            
            # Step 5: Validate evidence
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
            
            # Step 6: Fit to token budget
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
        
        # Step 7: Build optimized prompt
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
        )
        
        # Step 8: Generate response
        result = await self.lm_client.chat(messages, stream=request.stream)
        
        if request.stream:
            return self._wrap_stream_optimized(request, result, retrieved, background_tasks), retrieved
        
        # Step 9: Update state and memory
        text = result
        await self._post_turn_optimized(request, text, background_tasks)
        return text, retrieved

    async def _handle_chat_legacy(
        self, request: ChatRequest, background_tasks: BackgroundTasks | None = None
    ) -> Tuple[str | AsyncGenerator[str, None], List[RetrievalHit]]:
        """
        Legacy chat pipeline (backward compatible).
        """
        history = self.session_buffer.get(request.conversation_id)
        
        needs_rag = _needs_rag_retrieval(request.message)
        logger.debug(f"[Legacy] Message needs RAG: {needs_rag}")
        
        client_id = request.client_id
        if not client_id and needs_rag:
            client_id = await self._detect_client(request.message, request.conversation_id)
        
        retrieved = []
        memory_hits = []
        
        if needs_rag and request.top_k > 0:
            retrieved = self.retriever.search(
                query=request.message, top_k=request.top_k, metadata_filters=request.metadata_filters
            )
            
            if client_id:
                client_store = get_client_vector_store(client_id)
                client_hits = client_store.query(
                    query=request.message, top_k=request.top_k, collection="documents"
                )
                retrieved = client_hits + retrieved
            else:
                all_clients = await self.client_store.list_all()
                all_client_hits = []
                for client in all_clients:
                    client_store = get_client_vector_store(client.id)
                    if client_store.docs.count() > 0:
                        hits = client_store.query(
                            query=request.message, top_k=request.top_k, collection="documents"
                        )
                        for hit in hits:
                            hit.metadata["client_name"] = client.name
                        all_client_hits.extend(hits)
                
                if all_client_hits:
                    all_client_hits.sort(key=lambda x: x.score)
                    retrieved = all_client_hits[:request.top_k] + retrieved
            
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
        use_optimized_pipeline=use_optimized,
    )
