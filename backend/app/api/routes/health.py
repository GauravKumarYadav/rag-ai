from fastapi import APIRouter, Depends, HTTPException

from app.clients.lmstudio import LMStudioClient, get_lmstudio_client

router = APIRouter()


@router.get("", summary="Check backend and LM Studio connectivity")
async def health(client: LMStudioClient = Depends(get_lmstudio_client)) -> dict:
    ok = await client.healthcheck()
    if not ok:
        raise HTTPException(status_code=503, detail="LM Studio is not reachable")
    return {"status": "ok"}

