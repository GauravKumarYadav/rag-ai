import json
from typing import Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from app.models.schemas import ChatRequest, ImageInput
from app.services.chat_service import ChatService, get_chat_service
from app.auth.jwt import decode_token

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections."""
    
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)


manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    return manager


@router.websocket("/ws/{client_id}")
async def websocket_chat(
    websocket: WebSocket,
    client_id: str,
    token: str = Query(..., description="JWT authentication token"),
    service: ChatService = Depends(get_chat_service),
):
    """
    WebSocket endpoint for real-time chat.
    
    Requires authentication via token query parameter: /chat/ws/{client_id}?token=<jwt_token>
    
    Expected message format:
    {
        "type": "chat",
        "conversation_id": "optional-id",
        "message": "user message",
        "images": [{"data": "base64...", "media_type": "image/png"}],
        "top_k": 4,
        "system_prompt": "optional custom prompt"
    }
    
    Response format:
    - For streaming: {"type": "chunk", "content": "..."} followed by {"type": "done", "sources": [...]}
    - For errors: {"type": "error", "detail": "..."}
    """
    # Verify JWT token before accepting connection
    payload = decode_token(token)
    if payload is None:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return
    
    user_id = payload.get("sub")
    if user_id is None:
        await websocket.close(code=4001, reason="Invalid token payload")
        return
    
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            msg_type = data.get("type", "chat")
            
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            if msg_type != "chat":
                await websocket.send_json({
                    "type": "error",
                    "detail": f"Unknown message type: {msg_type}"
                })
                continue
            
            # Parse chat request
            message = data.get("message")
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "detail": "message is required"
                })
                continue
            
            images = [
                ImageInput(data=img["data"], media_type=img.get("media_type", "image/png"))
                for img in data.get("images", [])
            ]
            
            request = ChatRequest(
                conversation_id=data.get("conversation_id", client_id),
                client_id=client_id,
                message=message,
                images=images,
                stream=True,  # Always stream over WebSocket
                top_k=data.get("top_k", 4),
                include_sources=data.get("include_sources", True),
                metadata_filters=data.get("metadata_filters"),
                system_prompt=data.get("system_prompt"),
            )
            
            try:
                result, sources = await service.handle_chat(request, background_tasks=None)
                
                # Stream plain text chunks directly
                full_response = ""
                async for chunk in result:
                    if chunk:
                        full_response += chunk
                        await websocket.send_json({
                            "type": "chunk",
                            "content": chunk
                        })
                
                # Send completion with sources
                await websocket.send_json({
                    "type": "done",
                    "response": full_response,
                    "sources": [s.model_dump() for s in sources] if sources else []
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "detail": str(e)
                })
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        manager.disconnect(client_id)
        raise
