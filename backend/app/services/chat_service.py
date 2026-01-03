from typing import AsyncGenerator, List, Optional, Tuple

from fastapi import BackgroundTasks

from app.clients.lmstudio import LMStudioClient, get_lmstudio_client
from app.memory.long_term import LongTermMemory, get_long_term_memory
from app.memory.session_buffer import SessionBuffer, get_session_buffer
from app.models.client import ClientStore
from app.models.schemas import ChatRequest, RetrievalHit
from app.rag.retriever import Retriever, get_retriever
from app.rag.vector_store import get_client_vector_store
from app.services.client_extractor import ClientExtractor, get_client_extractor
from app.services.prompt_builder import build_messages


class ChatService:
    def __init__(
        self,
        lm_client: LMStudioClient,
        retriever: Retriever,
        session_buffer: SessionBuffer,
        long_term: LongTermMemory,
        client_extractor: Optional[ClientExtractor] = None,
        client_store: Optional[ClientStore] = None,
    ) -> None:
        self.lm_client = lm_client
        self.retriever = retriever
        self.session_buffer = session_buffer
        self.long_term = long_term
        self.client_extractor = client_extractor or get_client_extractor()
        self.client_store = client_store or ClientStore()

    def _build_system_context(self) -> str:
        """Build context about the system's clients and documents for the LLM."""
        clients = self.client_store.list_all()
        
        if not clients:
            return "\n\n[System Info: No clients registered. No documents available.]"
        
        context_parts = ["\n\n[System Information:"]
        context_parts.append(f"Total clients: {len(clients)}")
        
        for client in clients:
            client_store = get_client_vector_store(client.id)
            chunk_count = client_store.docs.count()
            memory_count = client_store.memories.count()
            
            # Get unique document sources for this client
            doc_sources = []
            if chunk_count > 0:
                try:
                    # Get all documents to extract unique sources
                    all_docs = client_store.docs.get(include=["metadatas"])
                    sources = set()
                    for meta in all_docs.get("metadatas", []):
                        if meta and meta.get("source"):
                            sources.add(meta["source"])
                    doc_sources = list(sources)
                except Exception:
                    pass
            
            # Report actual document count (unique files), not chunk count
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
        
        # Try to find by ID first
        if client_id:
            client = self.client_store.get(client_id)
            if client:
                return client.id
        
        # Search by name
        if client_name:
            matches = self.client_store.search(client_name)
            if matches:
                return matches[0].id
        
        return None

    async def handle_chat(
        self, request: ChatRequest, background_tasks: BackgroundTasks | None = None
    ) -> Tuple[str | AsyncGenerator[str, None], List[RetrievalHit]]:
        history = self.session_buffer.get(request.conversation_id)
        
        # Detect client from the message
        client_id = await self._detect_client(request.message, request.conversation_id)
        
        # Retrieve from global and client-specific sources
        retrieved = []
        if request.top_k > 0:
            # Global retrieval
            retrieved = self.retriever.search(
                query=request.message, top_k=request.top_k, metadata_filters=request.metadata_filters
            )
            
            # Client-specific retrieval
            if client_id:
                # Search only the specific client's documents
                client_store = get_client_vector_store(client_id)
                client_hits = client_store.query(
                    query=request.message, top_k=request.top_k, collection="documents"
                )
                # Prepend client-specific results (they're more relevant for this client)
                retrieved = client_hits + retrieved
            else:
                # No specific client detected - search ALL client documents
                # This ensures we find relevant context even without explicit client mention
                all_clients = self.client_store.list_all()
                all_client_hits = []
                for client in all_clients:
                    client_store = get_client_vector_store(client.id)
                    if client_store.docs.count() > 0:
                        hits = client_store.query(
                            query=request.message, top_k=request.top_k, collection="documents"
                        )
                        # Add client name to metadata for context
                        for hit in hits:
                            hit.metadata["client_name"] = client.name
                        all_client_hits.extend(hits)
                
                # Sort by score and take top_k best results across all clients
                if all_client_hits:
                    all_client_hits.sort(key=lambda x: x.score)
                    retrieved = all_client_hits[:request.top_k] + retrieved
        
        memory_hits = self.long_term.retrieve(query=request.message)
        
        # Build context: system info + current client context
        system_context = self._build_system_context()
        client_context = ""
        if client_id:
            client = self.client_store.get(client_id)
            if client:
                client_context = f"\n\n[Current client context: {client.name}]"
        
        # Include system context so LLM knows about available clients and documents
        system_prompt = (request.system_prompt or "") + system_context + client_context
        messages = build_messages(
            system_prompt=system_prompt,
            user_text=request.message,
            images=request.images,
            retrieved_docs=retrieved,
            memory_hits=memory_hits,
            session_messages=history,
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

    def _should_summarize(self, conversation_id: str) -> bool:
        history = self.session_buffer.get(conversation_id)
        return len(history) >= self.session_buffer.max_messages

    def _wrap_stream(
        self,
        request: ChatRequest,
        stream: AsyncGenerator[str, None],
        retrieved: List[RetrievalHit],
        background_tasks: BackgroundTasks | None,
    ) -> AsyncGenerator[str, None]:
        async def generator() -> AsyncGenerator[str, None]:
            buffer = ""
            async for chunk in stream:
                buffer += chunk
                # Encode newlines to preserve them in SSE (will be decoded on frontend)
                encoded_chunk = chunk.replace('\n', '\\n')
                yield f"data: {encoded_chunk}\n\n"
            await self._post_turn(request, buffer, background_tasks)

        return generator()


def get_chat_service() -> ChatService:
    return ChatService(
        lm_client=get_lmstudio_client(),
        retriever=get_retriever(),
        session_buffer=get_session_buffer(),
        long_term=get_long_term_memory(),
        client_extractor=get_client_extractor(),
        client_store=ClientStore(),
    )

