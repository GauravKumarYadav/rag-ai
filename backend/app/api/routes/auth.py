"""Authentication API endpoints for login."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.auth.jwt import create_access_token
from app.auth.password import verify_password
from app.auth.dependencies import get_current_user
from app.db.mysql import get_user_by_username


router = APIRouter()


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserInfo(BaseModel):
    """Basic user info response."""
    user_id: str
    username: str
    is_superuser: bool


@router.post("/login", response_model=TokenResponse, summary="User login")
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT access token.
    
    Use the returned access_token in the Authorization header:
    `Authorization: Bearer <access_token>`
    
    For WebSocket connections, pass the token as a query parameter:
    `/chat/ws/{client_id}?token=<access_token>`
    """
    # Get user from database
    user = await get_user_by_username(request.username)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    from app.config import settings
    
    access_token = create_access_token(
        user_id=user["id"],
        username=user["username"],
        is_superuser=user.get("is_superuser", False),
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


@router.get("/me", response_model=UserInfo, summary="Get current user info")
async def get_me(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get information about the currently authenticated user."""
    return UserInfo(
        user_id=current_user["user_id"],
        username=current_user["username"],
        is_superuser=current_user["is_superuser"],
    )
