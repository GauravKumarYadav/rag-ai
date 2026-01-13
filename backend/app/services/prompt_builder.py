"""
Prompt Builder for Small Model RAG Optimization.

Enhanced to support:
- State block injection (structured conversation state)
- Compressed facts format (dense bullet points with citations)
- Running summary for older history
- Evidence disclaimers
"""

from typing import List, Optional, Union

from app.models.schemas import ChatMessage, ImageInput, RetrievalHit
from app.services.context_compressor import CompressedFact

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful personal document assistant. "
    "The user has uploaded their own documents (financial records, identity documents, personal files) "
    "and is asking YOU to help them understand and analyze THEIR OWN data. "
    "You have full permission to discuss, summarize, and explain any document content provided in the context. "
    "\n\n"
    "IMPORTANT GUIDELINES:\n"
    "- For casual greetings (hi, hello, how are you), respond naturally and conversationally. "
    "Do NOT list clients or documents unless specifically asked.\n"
    "- Only mention client information or document contents when the user asks about them, "
    "or when the retrieved context is directly relevant to their question.\n"
    "- If you cite sources, reference their ids.\n"
    "- Be concise for simple questions, detailed for complex ones.\n"
    "- If evidence is marked as low confidence, acknowledge uncertainty."
)

# Optimized prompt for small models - more explicit instructions
SMALL_MODEL_SYSTEM_PROMPT = (
    "You are a helpful document assistant. Answer based ONLY on the provided context.\n\n"
    "RULES:\n"
    "1. Use ONLY information from the 'Context' section below.\n"
    "2. If information is not in context, say 'I don't have that information.'\n"
    "3. Cite sources using [source_name] format.\n"
    "4. Keep answers concise and focused.\n"
    "5. For greetings, respond naturally without mentioning documents."
)


def _format_hits(hits: List[RetrievalHit], label: str) -> str:
    """Format retrieval hits in traditional format (backward compatible)."""
    if not hits:
        return ""
    lines = [label]
    for hit in hits:
        source = hit.metadata.get("source") or hit.id
        lines.append(f"[{source}] {hit.content}")
    return "\n".join(lines)


def _format_compressed_facts(facts: List[CompressedFact], label: str = "Context:") -> str:
    """Format compressed facts as dense bullet points with citations."""
    if not facts:
        return ""
    lines = [label]
    for fact in facts:
        lines.append(fact.to_citation_string())
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
    Build messages for LLM chat (backward compatible).
    
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


def build_optimized_messages(
    user_text: str,
    state_block: Optional[str] = None,
    compressed_facts: Optional[List[CompressedFact]] = None,
    recent_messages: Optional[List[ChatMessage]] = None,
    running_summary: Optional[str] = None,
    episodic_memories: Optional[List[str]] = None,
    evidence_disclaimer: Optional[str] = None,
    images: Optional[List[ImageInput]] = None,
    system_prompt: Optional[str] = None,
    provider: Union[str, object] = "ollama",
) -> List[dict]:
    """
    Build optimized messages for small model RAG.
    
    This is the recommended builder for small models, using:
    - State block instead of full chat history
    - Compressed facts instead of raw chunks
    - Running summary for older history
    - Evidence disclaimers for low confidence
    
    Args:
        user_text: The user's message
        state_block: Structured conversation state (replaces history)
        compressed_facts: Dense bullet facts with citations
        recent_messages: Only last few turns (sliding window)
        running_summary: Summary of older conversation
        episodic_memories: Important decisions/facts to remember
        evidence_disclaimer: Disclaimer if evidence is weak
        images: Optional images
        system_prompt: Optional system prompt override
        provider: LLM provider
        
    Returns:
        Optimized message list for small models
    """
    messages: List[dict] = []
    
    # System prompt
    base_prompt = system_prompt or SMALL_MODEL_SYSTEM_PROMPT
    messages.append({"role": "system", "content": base_prompt})
    
    # State block (replaces full history)
    if state_block:
        messages.append({"role": "system", "content": state_block})
    
    # Running summary of older history
    if running_summary:
        messages.append({
            "role": "system", 
            "content": f"[Previous conversation summary]\n{running_summary}"
        })
    
    # Episodic memories (important facts to remember)
    if episodic_memories:
        memory_text = "\n".join(f"- {m}" for m in episodic_memories)
        messages.append({
            "role": "system",
            "content": f"[Important facts from conversation]\n{memory_text}"
        })
    
    # Compressed facts (dense context)
    if compressed_facts:
        context_text = _format_compressed_facts(compressed_facts, "Context:")
        messages.append({"role": "system", "content": context_text})
    
    # Evidence disclaimer
    if evidence_disclaimer:
        messages.append({
            "role": "system",
            "content": f"[Note: {evidence_disclaimer}]"
        })
    
    # Recent messages (sliding window)
    if recent_messages:
        for message in recent_messages:
            messages.append(message.model_dump())
    
    # User message
    provider_str = str(provider).lower() if provider else "ollama"
    is_ollama = "ollama" in provider_str
    
    if images:
        if is_ollama:
            user_message = {
                "role": "user",
                "content": user_text,
                "images": [img.data for img in images],
            }
        else:
            user_content: List[dict] = [{"type": "text", "text": user_text}]
            for image in images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.media_type};base64,{image.data}"},
                })
            user_message = {"role": "user", "content": user_content}
        messages.append(user_message)
    else:
        messages.append({"role": "user", "content": user_text})
    
    return messages


def estimate_prompt_tokens(messages: List[dict]) -> int:
    """
    Estimate token count for a message list.
    
    Uses simple character-based heuristic (~4 chars/token).
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    total_chars += len(item.get("text", ""))
    
    return int(total_chars / 4)

