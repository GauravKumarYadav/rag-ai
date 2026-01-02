from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.clients.lmstudio import (
    get_lmstudio_client,
    get_current_provider_info,
    switch_llm_provider,
)
from app.clients.base import BaseLLMClient, LLMProvider
from app.config import settings


router = APIRouter()


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ProviderInfo(BaseModel):
    provider: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    context_window: int


class SwitchProviderRequest(BaseModel):
    provider: str  # "lmstudio", "ollama", "openai", "custom"
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None


class SwitchProviderResponse(BaseModel):
    success: bool
    provider: str
    model: str
    message: str


@router.get("", response_model=ModelsResponse, summary="List available models")
async def list_models(client: BaseLLMClient = Depends(get_lmstudio_client)):
    """List all models available from the current LLM provider."""
    try:
        models = await client.list_models()
        return ModelsResponse(
            data=[
                ModelInfo(
                    id=m.get("id", m.get("name", "unknown")),
                    object=m.get("object", "model"),
                    owned_by=m.get("owned_by", "local"),
                )
                for m in models
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch models: {str(e)}")


@router.get("/current", response_model=ProviderInfo, summary="Get current provider config")
async def get_current_provider():
    """Get the currently configured LLM provider and settings."""
    info = get_current_provider_info()
    return ProviderInfo(
        provider=str(info.get("provider", "unknown")),
        base_url=info.get("base_url", ""),
        model=info.get("model", ""),
        temperature=info.get("temperature", 0.7),
        max_tokens=info.get("max_tokens", 1024),
        context_window=settings.context_window,
    )


@router.post("/switch", response_model=SwitchProviderResponse, summary="Switch LLM provider")
async def switch_provider(request: SwitchProviderRequest):
    """
    Switch to a different LLM provider at runtime.
    
    Supported providers:
    - **lmstudio**: Local LMStudio server (default: localhost:1234)
    - **ollama**: Local Ollama server (default: localhost:11434)
    - **openai**: OpenAI API (requires api_key)
    - **custom**: Any OpenAI-compatible endpoint
    """
    valid_providers = ["lmstudio", "ollama", "openai", "custom"]
    if request.provider not in valid_providers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider. Must be one of: {valid_providers}"
        )
    
    try:
        client = switch_llm_provider(
            provider=request.provider,
            base_url=request.base_url,
            model=request.model,
            api_key=request.api_key,
        )
        
        # Test the connection
        healthy = await client.healthcheck()
        if not healthy:
            return SwitchProviderResponse(
                success=False,
                provider=request.provider,
                model=getattr(client, "model", "unknown"),
                message=f"Switched to {request.provider} but health check failed. Server may be unavailable."
            )
        
        return SwitchProviderResponse(
            success=True,
            provider=request.provider,
            model=getattr(client, "model", "unknown"),
            message=f"Successfully switched to {request.provider}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch provider: {str(e)}")


@router.get("/providers", summary="List supported providers")
async def list_providers():
    """List all supported LLM providers and their default configurations."""
    return {
        "providers": [
            {
                "id": "lmstudio",
                "name": "LMStudio",
                "description": "Local LMStudio server with OpenAI-compatible API",
                "default_url": settings.lmstudio_base_url,
                "default_model": settings.lmstudio_model,
                "requires_api_key": False,
            },
            {
                "id": "ollama",
                "name": "Ollama",
                "description": "Local Ollama server for running open-source models",
                "default_url": settings.ollama_base_url,
                "default_model": settings.ollama_model,
                "requires_api_key": False,
            },
            {
                "id": "openai",
                "name": "OpenAI",
                "description": "OpenAI API (GPT-4, GPT-3.5, etc.)",
                "default_url": settings.openai_base_url,
                "default_model": settings.openai_model,
                "requires_api_key": True,
            },
            {
                "id": "custom",
                "name": "Custom Endpoint",
                "description": "Any OpenAI-compatible API endpoint",
                "default_url": settings.custom_base_url,
                "default_model": settings.custom_model,
                "requires_api_key": False,
            },
        ]
    }
