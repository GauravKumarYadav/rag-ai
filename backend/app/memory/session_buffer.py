from collections import defaultdict, deque
from functools import lru_cache
from typing import Deque, Dict, Iterable, List

from app.config import settings
from app.models.schemas import ChatMessage


class SessionBuffer:
    def __init__(self, max_tokens: int, max_messages: int) -> None:
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.buffers: Dict[str, Deque[ChatMessage]] = defaultdict(deque)

    def add(self, conversation_id: str, role: str, content: str) -> None:
        buffer = self.buffers[conversation_id]
        buffer.append(ChatMessage(role=role, content=content))
        self._trim(buffer)

    def get(self, conversation_id: str) -> List[ChatMessage]:
        return list(self.buffers.get(conversation_id, []))

    def _trim(self, buffer: Deque[ChatMessage]) -> None:
        while len(buffer) > self.max_messages:
            buffer.popleft()
        while self._estimated_tokens(buffer) > self.max_tokens and buffer:
            buffer.popleft()

    def _estimated_tokens(self, messages: Iterable[ChatMessage]) -> int:
        return sum(len(message.content.split()) for message in messages)


@lru_cache(maxsize=1)
def get_session_buffer() -> SessionBuffer:
    return SessionBuffer(
        max_tokens=settings.session_max_tokens,
        max_messages=settings.session_max_messages,
    )

