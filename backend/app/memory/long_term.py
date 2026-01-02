import time
from typing import List

from app.clients.lmstudio import LMStudioClient, get_lmstudio_client
from app.models.schemas import ChatMessage, RetrievalHit
from app.rag.vector_store import VectorStore, get_vector_store


SUMMARY_SYSTEM_PROMPT = (
    "You are a concise assistant that summarizes conversations for long-term recall. "
    "Capture key facts, user preferences, and unresolved items in bullet form."
)


class LongTermMemory:
    def __init__(self, store: VectorStore, lm_client: LMStudioClient) -> None:
        self.store = store
        self.lm_client = lm_client

    async def summarize_and_store(self, conversation_id: str, messages: List[ChatMessage]) -> RetrievalHit | None:
        if not messages:
            return None

        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages)
        prompt_messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this conversation:\n{transcript}"},
        ]
        summary = await self.lm_client.chat(prompt_messages, stream=False)
        doc_id = f"{conversation_id}-{int(time.time())}"
        metadata = {"conversation_id": conversation_id, "type": "summary"}
        self.store.add_memories(contents=[summary], ids=[doc_id], metadatas=[metadata])
        return RetrievalHit(id=doc_id, content=summary, score=0.0, metadata=metadata)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalHit]:
        return self.store.query(query=query, top_k=top_k, collection="memories")


def get_long_term_memory() -> LongTermMemory:
    return LongTermMemory(store=get_vector_store(), lm_client=get_lmstudio_client())

