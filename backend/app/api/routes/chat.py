import base64
import json
from typing import Any, Dict, List, Optional, Set

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
from app.auth.dependencies import get_allowed_clients, GLOBAL_CLIENT_ID

router = APIRouter()


def validate_and_get_client_id(
    requested_client_id: Optional[str],
    allowed_clients: Set[str],
    user_id: str,
) -> str:
    """
    Validate client access and return the effective client ID.
    
    Args:
        requested_client_id: The client_id from the request (may be None)
        allowed_clients: Set of client IDs the user can access
        user_id: The user's ID for logging
        
    Returns:
        The validated client_id (defaults to 'global' if not specified)
        
    Raises:
        HTTPException 403 if user doesn't have access to requested client
    """
    # Default to global client if not specified
    if requested_client_id is None:
        return GLOBAL_CLIENT_ID
    
    # Validate access
    if requested_client_id not in allowed_clients:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied to client '{requested_client_id}'. "
                   f"Your allowed clients: {sorted(allowed_clients)}",
        )
    
    return requested_client_id


@router.post("", summary="Chat with multimodal support")
async def chat(
    request: Request,
    background_tasks: BackgroundTasks,
    service: ChatService = Depends(get_chat_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    allowed_clients: Set[str] = Depends(get_allowed_clients),
):
    """
    Chat endpoint supporting both JSON and form-data.
    
    JSON body: {"message": "...", "conversation_id": "...", "stream": true/false, ...}
    Form data: message, conversation_id, stream, files[], etc.
    
    Client Access:
    - If client_id is provided, it must be in the user's allowed_clients
    - If client_id is omitted, defaults to 'global' client
    - Use the /clients endpoint to see available clients
    """
    content_type = request.headers.get("content-type", "")
    user_id = current_user.get("user_id", "unknown")
    
    if "application/json" in content_type:
        # Handle JSON request
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        
        if not body.get("message"):
            raise HTTPException(status_code=400, detail="message is required")
        
        # Validate client access
        requested_client_id = body.get("client_id")
        validated_client_id = validate_and_get_client_id(
            requested_client_id,
            allowed_clients,
            user_id,
        )
        
        chat_request = ChatRequest(
            conversation_id=body.get("conversation_id", "default"),
            message=body["message"],
            images=[ImageInput(**img) for img in body.get("images", [])],
            stream=body.get("stream", True),
            top_k=body.get("top_k", 4),
            include_sources=body.get("include_sources", True),
            metadata_filters=body.get("metadata_filters"),
            system_prompt=body.get("system_prompt"),
            client_id=validated_client_id,
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
        
        # Validate client access
        requested_client_id = form.get("client_id")
        validated_client_id = validate_and_get_client_id(
            requested_client_id,
            allowed_clients,
            user_id,
        )
        
        chat_request = ChatRequest(
            conversation_id=form.get("conversation_id") or "default",
            message=str(message),
            images=images,
            stream=str(form.get("stream", "true")).lower() == "true",
            top_k=int(form.get("top_k", 4)),
            include_sources=str(form.get("include_sources", "true")).lower() == "true",
            metadata_filters=metadata_filters,
            system_prompt=form.get("system_prompt"),
            client_id=validated_client_id,
        )
    
    # Pass allowed_clients to service for retrieval filtering
    result, sources = await service.handle_chat(
        chat_request, 
        background_tasks,
        allowed_clients=allowed_clients,
    )
    
    if chat_request.stream:
        async def sse_wrapper():
            """Wrap plain text chunks from chat service in SSE framing."""
            message_id = __import__("uuid").uuid4().hex[:8]
            start_time = __import__("time").monotonic()
            yield f"data: {json.dumps({'type': 'start', 'message_id': message_id})}\n\n"
            async for chunk in result:
                encoded = chunk.replace('\n', '\\n').replace('\r', '\\r')
                yield f"data: {json.dumps({'type': 'content', 'chunk': encoded})}\n\n"
            duration = __import__("time").monotonic() - start_time
            yield f"data: {json.dumps({'type': 'end', 'duration_ms': int(duration * 1000)})}\n\n"

        return StreamingResponse(sse_wrapper(), media_type="text/event-stream")
    
    payload = ChatResponse(response=result, sources=sources)
    return JSONResponse(content=payload.model_dump())
