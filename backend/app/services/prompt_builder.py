from typing import List

from app.models.schemas import ChatMessage, ImageInput, RetrievalHit

DEFAULT_SYSTEM_PROMPT = (
    "You are a local multimodal assistant. "
    "Use provided context, document snippets, and memories to answer accurately. "
    "If you cite sources, reference their ids."
)


def _format_hits(hits: List[RetrievalHit], label: str) -> str:
    if not hits:
        return ""
    lines = [label]
    for hit in hits:
        source = hit.metadata.get("source") or hit.id
        lines.append(f"[{source}] {hit.content}")
    return "\n".join(lines)


def build_messages(
    system_prompt: str | None,
    user_text: str,
    images: List[ImageInput],
    retrieved_docs: List[RetrievalHit],
    memory_hits: List[RetrievalHit],
    session_messages: List[ChatMessage],
) -> List[dict]:
    messages: List[dict] = []
    messages.append({"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT})

    memory_block = _format_hits(memory_hits, "Relevant memories:")
    if memory_block:
        messages.append({"role": "system", "content": memory_block})

    context_block = _format_hits(retrieved_docs, "Retrieved context:")
    if context_block:
        messages.append({"role": "system", "content": context_block})

    for message in session_messages:
        messages.append(message.model_dump())

    user_content: List[dict] = [{"type": "text", "text": user_text}]
    for image in images:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image.media_type};base64,{image.data}"},
            }
        )
    messages.append({"role": "user", "content": user_content})
    return messages

