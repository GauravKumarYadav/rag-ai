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
                client_store = get_client_vector_store(client_id)
                client_hits = client_store.query(
                    query=request.message, top_k=request.top_k, collection="documents"
                )
                # Prepend client-specific results (they're more relevant for this client)
                retrieved = client_hits + retrieved
        
        memory_hits = self.long_term.retrieve(query=request.message)
        
        # Build context message about current client
        client_context = ""
        if client_id:
            client = self.client_store.get(client_id)
            if client:
                client_context = f"\n\n[Current client context: {client.name}]"
        
        system_prompt = (request.system_prompt or "") + client_context
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

