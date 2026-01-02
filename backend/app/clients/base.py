"""
Abstract base class for LLM providers.
Supports LMStudio, Ollama, OpenAI, and custom endpoints.
"""
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncGenerator, List, Mapping, MutableMapping, Optional

import httpx


class LLMProvider(str, Enum):
    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"
    OPENAI = "openai"
    CUSTOM = "custom"


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def healthcheck(self) -> bool:
        """Check if the LLM service is available."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Mapping[str, Any]],
        stream: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None] | str:
        """Send a chat request to the LLM."""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[dict]:
        """List available models."""
        pass


class OpenAICompatibleClient(BaseLLMClient):
    """
    Client for OpenAI-compatible APIs.
    Works with: LMStudio, Ollama, OpenAI, vLLM, LocalAI, etc.
    """
    
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        provider: LLMProvider = LLMProvider.LMSTUDIO,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.provider = provider
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self.client = httpx.AsyncClient(timeout=timeout, headers=headers)
    
    async def healthcheck(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
    
    async def list_models(self) -> List[dict]:
        try:
            response = await self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            data = response.json()
            return data.get("data", data.get("models", []))
        except Exception:
            return []
    
    async def chat(
        self,
        messages: List[Mapping[str, Any]],
        stream: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None] | str:
        payload: MutableMapping[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": stream,
        }
        
        if stream:
            return self._stream_chat(payload)
        return await self._chat_once(payload)
    
    async def _chat_once(self, payload: Mapping[str, Any]) -> str:
        url = f"{self.base_url}/chat/completions"
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return message.get("content", "")
    
    async def _stream_chat(self, payload: Mapping[str, Any]) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/chat/completions"
        async with self.client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                clean = line
                if clean.startswith("data:"):
                    clean = clean.split("data:", 1)[1].strip()
                if clean in ("", "[DONE]"):
                    continue
                try:
                    event = json.loads(clean)
                except json.JSONDecodeError:
                    continue
                delta = event.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content


class OllamaClient(BaseLLMClient):
    """
    Native Ollama client with Ollama-specific features.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def healthcheck(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
    
    async def list_models(self) -> List[dict]:
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [{"id": m["name"], "name": m["name"]} for m in data.get("models", [])]
        except Exception:
            return []
    
    async def chat(
        self,
        messages: List[Mapping[str, Any]],
        stream: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None] | str:
        # Convert OpenAI format to Ollama format
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            },
        }
        
        if stream:
            return self._stream_chat(payload)
        return await self._chat_once(payload)
    
    async def _chat_once(self, payload: dict) -> str:
        payload["stream"] = False
        response = await self.client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")
    
    async def _stream_chat(self, payload: dict) -> AsyncGenerator[str, None]:
        async with self.client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue
