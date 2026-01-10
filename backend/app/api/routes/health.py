from fastapi import APIRouter, Depends, HTTPException

from app.clients.lmstudio import LMStudioClient, get_lmstudio_client, get_current_provider_info

router = APIRouter()


@router.get("", summary="Check backend and LLM connectivity")
async def health(client: LMStudioClient = Depends(get_lmstudio_client)) -> dict:
    """Health check with active LLM provider information.
    
    Returns:
        status: "ok" if LLM is reachable
        llm_provider: Active provider (lmstudio, ollama, openai, custom)
        llm_model: Currently configured model name
        llm_base_url: API endpoint being used
    """
    ok = await client.healthcheck()
    provider_info = get_current_provider_info()
    
    if not ok:
        raise HTTPException(
            status_code=503, 
            detail=f"{provider_info['provider']} is not reachable at {provider_info['base_url']}"
        )
    
    return {
        "status": "ok",
        "llm_provider": str(provider_info["provider"]).lower().replace("llmprovider.", ""),
        "llm_model": provider_info["model"],
        "llm_base_url": provider_info["base_url"],
    }

