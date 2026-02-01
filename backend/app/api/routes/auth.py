"""Authentication API endpoints for login."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.auth.jwt import create_access_token
from app.auth.password import verify_password
from app.auth.dependencies import get_current_user
from app.auth.users import get_user_by_username, get_user_client_ids, get_user_by_id, update_user_password
from app.config import settings


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
    id: str  # User ID
    user_id: str  # Alias for compatibility
    username: str
    email: str | None = None
    is_superuser: bool
    is_active: bool = True
    created_at: str | None = None


@router.post("/login", response_model=TokenResponse, summary="User login")
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT access token.
    
    Use the returned access_token in the Authorization header:
    `Authorization: Bearer <access_token>`
    
    For WebSocket connections, pass the token as a query parameter:
    `/chat/ws/{client_id}?token=<access_token>`
    
    Default admin credentials:
    - Username: admin
    - Password: admin123
    """
    # Get user from Redis-backed store
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
    
    # Fetch user's allowed clients for embedding in JWT
    allowed_clients = []
    if not user.get("is_superuser", False):
        # Regular users get their assigned client IDs
        allowed_clients = await get_user_client_ids(user["id"])
    # Superusers get "*" which means all clients (handled in dependency)
    
    access_token = create_access_token(
        user_id=user["id"],
        username=user["username"],
        is_superuser=user.get("is_superuser", False),
        allowed_clients=allowed_clients,
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
    # Get full user data from store for additional fields
    user_data = await get_user_by_id(current_user["user_id"])
    
    return UserInfo(
        id=current_user["user_id"],
        user_id=current_user["user_id"],
        username=current_user["username"],
        email=user_data.get("email") if user_data else None,
        is_superuser=current_user["is_superuser"],
        is_active=user_data.get("is_active", True) if user_data else True,
        created_at=user_data.get("created_at") if user_data else None,
    )


class ChangePasswordRequest(BaseModel):
    """Change password request model."""
    current_password: str
    new_password: str


@router.post("/change-password", summary="Change password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Change the current user's password.
    
    Requires the current password for verification.
    """
    # Get full user data to verify current password
    user = await get_user_by_id(current_user["user_id"])
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Verify current password
    if not verify_password(request.current_password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    
    # Validate new password
    if len(request.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 6 characters",
        )
    
    # Update password
    success = await update_user_password(current_user["user_id"], request.new_password)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password",
        )
    
    return {"message": "Password changed successfully"}
