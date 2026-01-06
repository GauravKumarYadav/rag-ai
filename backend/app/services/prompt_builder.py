from typing import List, Union

from app.models.schemas import ChatMessage, ImageInput, RetrievalHit

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful personal document assistant. "
    "The user has uploaded their own documents (financial records, identity documents, personal files) "
    "and is asking YOU to help them understand and analyze THEIR OWN data. "
    "You have full permission to discuss, summarize, and explain any document content provided in the context. "
    "Use the provided context, document snippets, and memories to answer accurately and helpfully. "
    "If you cite sources, reference their ids. "
    "Always be helpful and provide detailed explanations when asked about document contents."
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
    provider: Union[str, object] = "ollama",
) -> List[dict]:
    """
    Build messages for LLM chat.
    
    Args:
        system_prompt: Optional system prompt
        user_text: The user's message text
        images: List of images to include
        retrieved_docs: Retrieved document chunks for RAG
        memory_hits: Retrieved memory chunks
        session_messages: Previous conversation messages
        provider: LLM provider - "ollama" uses native Ollama format (images as list),
                  other providers use OpenAI vision format (structured content)
    
    Returns:
        List of message dicts formatted for the specified provider
    """
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

    # Build the user message based on provider format
    # Normalize provider to string for comparison (handles enum or string)
    provider_str = str(provider).lower() if provider else "ollama"
    is_ollama = "ollama" in provider_str
    
    if images:
        if is_ollama:
            # Ollama format: images as a list of base64 strings in "images" field
            # Works with vision models like llava, qwen-vl, llama3.2-vision, etc.
            # https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-images
            user_message = {
                "role": "user",
                "content": user_text,
                "images": [img.data for img in images],  # Just base64 data, no prefix
            }
        else:
            # OpenAI format: structured content with image_url objects
            # Used by OpenAI, Azure OpenAI, LMStudio, etc.
            user_content: List[dict] = [{"type": "text", "text": user_text}]
            for image in images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image.media_type};base64,{image.data}"},
                    }
                )
            user_message = {"role": "user", "content": user_content}
        messages.append(user_message)
    else:
        # Text-only: simple string content (works for all providers)
        messages.append({"role": "user", "content": user_text})
    
    return messages

