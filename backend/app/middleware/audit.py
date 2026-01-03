"""Audit logging middleware for tracking all API requests."""

import asyncio
import logging
import time
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.auth.jwt import decode_token
from app.db.mysql import log_audit_event

logger = logging.getLogger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all requests to the MySQL audit_logs table.
    
    Captures:
    - User identity (from JWT token if present)
    - Client ID (from header or path)
    - Request method, path, status code
    - IP address and user agent
    - Request duration in milliseconds
    """
    
    # Routes that should be excluded from audit logging
    EXCLUDED_PATHS = {
        "/health",
        "/health/live",
        "/health/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and log audit information."""
        
        # Skip logging for excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)
        
        start_time = time.perf_counter()
        
        # Extract user info from JWT token if present
        user_id: Optional[str] = None
        username: Optional[str] = None
        
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = decode_token(token)
            if payload:
                user_id = payload.get("sub")
                username = payload.get("username")
        
        # Extract client_id from various sources
        client_id = self._extract_client_id(request)
        
        # Get client IP (handle proxy headers)
        ip_address = self._get_client_ip(request)
        
        # Get user agent
        user_agent = request.headers.get("User-Agent", "")[:500]  # Limit length
        
        # Determine action from method and path
        action = self._determine_action(request.method, request.url.path)
        
        # Process the request
        response: Response
        try:
            response = await call_next(request)
        except Exception as e:
            # Log failed requests too
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            asyncio.create_task(
                log_audit_event(
                    user_id=user_id,
                    username=username,
                    client_id=client_id,
                    action=action,
                    resource=request.url.path.split("/")[1] if "/" in request.url.path else None,
                    method=request.method,
                    path=str(request.url.path),
                    status_code=500,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    duration_ms=duration_ms,
                    request_summary=f"Error: {str(e)[:200]}",
                )
            )
            raise
        
        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Log the audit event asynchronously (fire and forget)
        asyncio.create_task(
            log_audit_event(
                user_id=user_id,
                username=username,
                client_id=client_id,
                action=action,
                resource=request.url.path.split("/")[1] if "/" in request.url.path else None,
                method=request.method,
                path=str(request.url.path),
                status_code=response.status_code,
                ip_address=ip_address,
                user_agent=user_agent,
                duration_ms=duration_ms,
                request_summary=None,
            )
        )
        
        return response
    
    def _extract_client_id(self, request: Request) -> Optional[str]:
        """Extract client_id from header, path, or query params."""
        
        # Try X-Client-ID header first
        client_id = request.headers.get("X-Client-ID")
        if client_id:
            return client_id
        
        # Try extracting from path (e.g., /chat/ws/{client_id})
        path_parts = request.url.path.strip("/").split("/")
        if len(path_parts) >= 3 and path_parts[0] == "chat" and path_parts[1] == "ws":
            return path_parts[2]
        
        # Try extracting from path (e.g., /conversations/{conversation_id})
        if len(path_parts) >= 2 and path_parts[0] == "conversations":
            return path_parts[1]
        
        # Try query params
        client_id = request.query_params.get("client_id")
        if client_id:
            return client_id
        
        return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address, handling proxy headers."""
        
        # Check X-Forwarded-For header (set by proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the first IP in the chain (original client)
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _determine_action(self, method: str, path: str) -> str:
        """Determine the action type from method and path."""
        
        path_lower = path.lower()
        
        # Specific action types based on path patterns
        if "/auth/login" in path_lower:
            return "login"
        if "/auth/register" in path_lower:
            return "register"
        if "/chat" in path_lower:
            return "chat"
        if "/documents/upload" in path_lower:
            return "document_upload"
        if "/documents/search" in path_lower:
            return "document_search"
        if "/conversations" in path_lower:
            return "conversation_access"
        if "/clients" in path_lower:
            return "client_management"
        
        # Generic action based on HTTP method
        method_actions = {
            "GET": "read",
            "POST": "create",
            "PUT": "update",
            "PATCH": "update",
            "DELETE": "delete",
        }
        
        return method_actions.get(method.upper(), "request")
