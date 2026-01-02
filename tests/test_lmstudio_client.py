import pytest
import respx
from httpx import Response

from app.clients.lmstudio import LMStudioClient


@pytest.mark.asyncio
@respx.mock
async def test_healthcheck_ok() -> None:
    respx.get("http://localhost:1234/v1/models").mock(return_value=Response(200))
    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        model="test-model",
        temperature=0.2,
        max_tokens=128,
    )
    assert await client.healthcheck()


@pytest.mark.asyncio
@respx.mock
async def test_stream_chat_parses_chunks() -> None:
    body = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=Response(
            status_code=200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )
    )
    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        model="test-model",
        temperature=0.2,
        max_tokens=128,
    )
    stream = await client.chat([{"role": "user", "content": "hi"}], stream=True)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    assert chunks == ["Hello"]

