from app.models.schemas import ChatMessage, ImageInput, RetrievalHit
from app.services.prompt_builder import DEFAULT_SYSTEM_PROMPT, build_messages


def test_prompt_builder_includes_images() -> None:
    messages = build_messages(
        system_prompt=None,
        user_text="hello",
        images=[ImageInput(data="abc", media_type="image/png")],
        retrieved_docs=[],
        memory_hits=[],
        session_messages=[],
    )
    user_parts = messages[-1]["content"]
    assert any(part["type"] == "image_url" for part in user_parts)


def test_prompt_builder_includes_context_blocks() -> None:
    retrieved = [RetrievalHit(id="doc1", content="fact", score=0.1, metadata={"source": "doc1"})]
    history = [ChatMessage(role="assistant", content="previous")]
    messages = build_messages(
        system_prompt=None,
        user_text="question",
        images=[],
        retrieved_docs=retrieved,
        memory_hits=[],
        session_messages=history,
    )
    system_messages = [m for m in messages if m["role"] == "system"]
    assert DEFAULT_SYSTEM_PROMPT in system_messages[0]["content"]
    assert "Retrieved context" in system_messages[-1]["content"]

