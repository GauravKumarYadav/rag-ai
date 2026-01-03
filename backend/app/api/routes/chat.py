import base64
import json
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse, ImageInput
from app.services.chat_service import ChatService
from app.dependencies import get_current_user, get_chat_service

router = APIRouter()


@router.post("", summary="Chat with multimodal support")
async def chat(
    request: Request,
    background_tasks: BackgroundTasks,
    service: ChatService = Depends(get_chat_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Chat endpoint supporting both JSON and form-data.
    
    JSON body: {"message": "...", "conversation_id": "...", "stream": true/false, ...}
    Form data: message, conversation_id, stream, files[], etc.
    """
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
        # Handle JSON request
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        
        if not body.get("message"):
            raise HTTPException(status_code=400, detail="message is required")
        
        chat_request = ChatRequest(
            conversation_id=body.get("conversation_id", "default"),
            message=body["message"],
            images=[ImageInput(**img) for img in body.get("images", [])],
            stream=body.get("stream", True),
            top_k=body.get("top_k", 4),
            include_sources=body.get("include_sources", True),
            metadata_filters=body.get("metadata_filters"),
            system_prompt=body.get("system_prompt"),
        )
    else:
        # Handle form-data request
        form = await request.form()
        message = form.get("message")
        
        if not message:
            raise HTTPException(status_code=400, detail="message is required")
        
        images: List[ImageInput] = []
        files = form.getlist("files")
        for file in files:
            if hasattr(file, "read"):
                content = await file.read()
                encoded = base64.b64encode(content).decode("utf-8")
                images.append(ImageInput(
                    data=encoded, 
                    media_type=getattr(file, "content_type", "image/png") or "image/png"
                ))
        
        metadata_filters = None
        if form.get("metadata_filters"):
            try:
                metadata_filters = json.loads(form.get("metadata_filters"))
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="metadata_filters must be valid JSON")
        
        chat_request = ChatRequest(
            conversation_id=form.get("conversation_id") or "default",
            message=str(message),
            images=images,
            stream=str(form.get("stream", "true")).lower() == "true",
            top_k=int(form.get("top_k", 4)),
            include_sources=str(form.get("include_sources", "true")).lower() == "true",
            metadata_filters=metadata_filters,
            system_prompt=form.get("system_prompt"),
        )
    
    result, sources = await service.handle_chat(chat_request, background_tasks)
    
    if chat_request.stream:
        return StreamingResponse(result, media_type="text/event-stream")
    
    payload = ChatResponse(response=result, sources=sources)
    return JSONResponse(content=payload.model_dump())

