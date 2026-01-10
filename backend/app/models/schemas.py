from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    data: str
    media_type: str = "image/png"


class RetrievalHit(BaseModel):
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryTrace(BaseModel):
    id: str
    summary: str
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    conversation_id: str = "default"
    message: str
    images: List[ImageInput] = Field(default_factory=list)
    stream: bool = True
    top_k: int = 4
    include_sources: bool = True
    metadata_filters: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    client_id: Optional[str] = None  # Explicit client ID for document retrieval


class ChatResponse(BaseModel):
    response: str
    sources: List[RetrievalHit] = Field(default_factory=list)

