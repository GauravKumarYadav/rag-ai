"""
LLM Client factory - supports multiple providers.
"""
import json
from functools import lru_cache
from typing import Any, AsyncGenerator, List, Mapping, MutableMapping, Optional

import httpx

from app.config import settings
from app.clients.base import (
    BaseLLMClient,
    LLMProvider,
    OllamaClient,
    OpenAICompatibleClient,
)


# Keep LMStudioClient as an alias for backwards compatibility
class LMStudioClient(OpenAICompatibleClient):
    """LMStudio client - wrapper around OpenAI-compatible client."""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            provider=LLMProvider.LMSTUDIO,
        )


def create_llm_client(
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> BaseLLMClient:
    """
    Factory function to create an LLM client based on provider.
    
    Args:
        provider: One of "lmstudio", "ollama", "openai", "custom"
        base_url: Override the default base URL
        model: Override the default model
        api_key: API key (for OpenAI or custom endpoints)
        temperature: Override default temperature
        max_tokens: Override default max_tokens
    
    Returns:
        An LLM client instance
    """
    provider = provider or settings.llm_provider
    temp = temperature if temperature is not None else settings.llm_temperature
    tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
    timeout = settings.llm_timeout
    
    if provider == "ollama":
        return OllamaClient(
            base_url=base_url or settings.ollama_base_url,
            model=model or settings.ollama_model,
            temperature=temp,
            max_tokens=tokens,
            timeout=timeout,
        )
    elif provider == "openai":
        return OpenAICompatibleClient(
            base_url=base_url or settings.openai_base_url,
            model=model or settings.openai_model,
            api_key=api_key or settings.openai_api_key,
            temperature=temp,
            max_tokens=tokens,
            timeout=timeout,
            provider=LLMProvider.OPENAI,
        )
    elif provider == "custom":
        return OpenAICompatibleClient(
            base_url=base_url or settings.custom_base_url,
            model=model or settings.custom_model,
            api_key=api_key or settings.custom_api_key,
            temperature=temp,
            max_tokens=tokens,
            timeout=timeout,
            provider=LLMProvider.CUSTOM,
        )
    else:  # Default to lmstudio
        return LMStudioClient(
            base_url=base_url or settings.lmstudio_base_url,
            model=model or settings.lmstudio_model,
            temperature=temp,
            max_tokens=tokens,
            timeout=timeout,
        )


# Global client instance (can be switched at runtime)
_current_client: Optional[BaseLLMClient] = None


def get_lmstudio_client() -> BaseLLMClient:
    """Get the current LLM client (creates one if needed)."""
    global _current_client
    if _current_client is None:
        _current_client = create_llm_client()
    return _current_client


def switch_llm_provider(
    provider: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseLLMClient:
    """
    Switch to a different LLM provider at runtime.
    
    Args:
        provider: One of "lmstudio", "ollama", "openai", "custom"
        base_url: Override the default base URL
        model: Override the default model  
        api_key: API key (for OpenAI or custom endpoints)
    
    Returns:
        The new LLM client instance
    """
    global _current_client
    _current_client = create_llm_client(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
    )
    return _current_client


def get_current_provider_info() -> dict:
    """Get information about the current LLM provider."""
    client = get_lmstudio_client()
    return {
        "provider": getattr(client, "provider", "unknown"),
        "base_url": getattr(client, "base_url", "unknown"),
        "model": getattr(client, "model", "unknown"),
        "temperature": getattr(client, "temperature", 0.7),
        "max_tokens": getattr(client, "max_tokens", 1024),
    }

