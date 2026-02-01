"""Client management API endpoints."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.models.client import (
    Client,
    ClientCreate,
    ClientUpdate,
    get_client_store,
)
from app.auth.dependencies import get_current_user, require_superuser, get_allowed_clients
from app.auth.users import (
    get_users_for_client,
    add_user_client,
    remove_user_client,
    get_user_by_id,
)


router = APIRouter()


class ClientResponse(BaseModel):
    """Response model for client."""
    id: str
    name: str
    aliases: List[str]
    metadata: dict
    created_at: str
    updated_at: str


class ClientListResponse(BaseModel):
    """Response model for client list."""
    clients: List[ClientResponse]
    total: int


class ClientStatsResponse(BaseModel):
    """Response model for client stats."""
    client_id: str
    name: str
    document_count: int
    memory_count: int


@router.post("", response_model=ClientResponse, summary="Create a new client")
async def create_client(
    data: ClientCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Create a new client for document organization."""
    store = get_client_store()
    
    # Check if client with same name exists
    existing = await store.get_by_name(data.name, fuzzy=False)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Client with name '{data.name}' already exists (ID: {existing.id})",
        )
    
    client = await store.create(data)
    return ClientResponse(
        id=client.id,
        name=client.name,
        aliases=client.aliases,
        metadata=client.metadata,
        created_at=client.created_at.isoformat(),
        updated_at=client.updated_at.isoformat(),
    )


@router.get("", response_model=ClientListResponse, summary="List all clients")
async def list_clients(
    search: Optional[str] = Query(None, description="Search by name"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List all clients, optionally filtered by search query."""
    store = get_client_store()
    
    if search:
        clients = await store.search(search)
    else:
        clients = await store.list_all()
    
    return ClientListResponse(
        clients=[
            ClientResponse(
                id=c.id,
                name=c.name,
                aliases=c.aliases,
                metadata=c.metadata,
                created_at=c.created_at.isoformat(),
                updated_at=c.updated_at.isoformat(),
            )
            for c in clients
        ],
        total=len(clients),
    )


@router.get("/my/assigned", response_model=ClientListResponse, summary="Get assigned clients")
async def get_my_assigned_clients(
    current_user: Dict[str, Any] = Depends(get_current_user),
    allowed_clients: set = Depends(get_allowed_clients),
):
    """
    Get clients assigned to the current user.
    
    Superusers get all clients. Regular users get only their assigned clients.
    """
    store = get_client_store()
    
    # Superusers get all clients
    if current_user.get("is_superuser"):
        clients = await store.list_all()
    else:
        # Regular users get only their assigned clients
        clients = []
        for client_id in allowed_clients:
            client = await store.get(client_id)
            if client:
                clients.append(client)
    
    return ClientListResponse(
        clients=[
            ClientResponse(
                id=c.id,
                name=c.name,
                aliases=c.aliases,
                metadata=c.metadata,
                created_at=c.created_at.isoformat(),
                updated_at=c.updated_at.isoformat(),
            )
            for c in clients
        ],
        total=len(clients),
    )


@router.get("/search/by-name", response_model=ClientResponse, summary="Find client by name")
async def find_client_by_name(
    name: str = Query(..., description="Client name to search"),
    fuzzy: bool = Query(True, description="Allow fuzzy matching"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Find a client by name (exact or fuzzy match)."""
    store = get_client_store()
    client = await store.get_by_name(name, fuzzy=fuzzy)
    
    if not client:
        raise HTTPException(status_code=404, detail=f"No client found matching '{name}'")
    
    return ClientResponse(
        id=client.id,
        name=client.name,
        aliases=client.aliases,
        metadata=client.metadata,
        created_at=client.created_at.isoformat(),
        updated_at=client.updated_at.isoformat(),
    )


@router.get("/{client_id}", response_model=ClientResponse, summary="Get client by ID")
async def get_client(
    client_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get a specific client by ID."""
    store = get_client_store()
    client = await store.get(client_id)
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return ClientResponse(
        id=client.id,
        name=client.name,
        aliases=client.aliases,
        metadata=client.metadata,
        created_at=client.created_at.isoformat(),
        updated_at=client.updated_at.isoformat(),
    )


@router.put("/{client_id}", response_model=ClientResponse, summary="Update client")
async def update_client(
    client_id: str,
    data: ClientUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Update a client's information."""
    store = get_client_store()
    client = await store.update(client_id, data)
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return ClientResponse(
        id=client.id,
        name=client.name,
        aliases=client.aliases,
        metadata=client.metadata,
        created_at=client.created_at.isoformat(),
        updated_at=client.updated_at.isoformat(),
    )


@router.delete("/{client_id}", summary="Delete client")
async def delete_client(
    client_id: str,
    current_user: Dict[str, Any] = Depends(require_superuser),
):
    """
    Delete a client.
    
    Requires admin privileges. The global client cannot be deleted.
    """
    store = get_client_store()
    client = await store.get(client_id)
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    if client_id == "global":
        raise HTTPException(status_code=400, detail="Cannot delete the global client")
    
    await store.delete(client_id)
    
    return {"message": f"Client '{client.name}' deleted"}


@router.get("/{client_id}/stats", response_model=ClientStatsResponse, summary="Get client stats")
async def get_client_stats(
    client_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get document and memory statistics for a client."""
    from app.rag.vector_store import get_client_vector_store
    
    client_store = get_client_store()
    client = await client_store.get(client_id)
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    try:
        vector_store = get_client_vector_store(client_id)
        stats = vector_store.get_stats()
        doc_count = stats.get("document_count", 0)
        mem_count = stats.get("memory_count", 0)
    except Exception:
        doc_count = 0
        mem_count = 0
    
    return ClientStatsResponse(
        client_id=client_id,
        name=client.name,
        document_count=doc_count,
        memory_count=mem_count,
    )


class UserAccessResponse(BaseModel):
    """Response model for user access info."""
    id: str
    username: str
    email: Optional[str] = None
    is_superuser: bool
    is_active: bool


class UserAccessListResponse(BaseModel):
    """Response model for user access list."""
    users: List[UserAccessResponse]
    total: int


@router.get("/{client_id}/users", response_model=UserAccessListResponse, summary="List users with access")
async def list_client_users(
    client_id: str,
    current_user: Dict[str, Any] = Depends(require_superuser),
):
    """
    List all users with access to a client.
    
    Requires admin privileges.
    """
    client_store = get_client_store()
    client = await client_store.get(client_id)
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    users = await get_users_for_client(client_id)
    
    return UserAccessListResponse(
        users=[
            UserAccessResponse(
                id=u["id"],
                username=u["username"],
                email=u.get("email"),
                is_superuser=u.get("is_superuser", False),
                is_active=u.get("is_active", True),
            )
            for u in users
        ],
        total=len(users),
    )


@router.post("/{client_id}/users/{user_id}", summary="Grant user access to client")
async def grant_user_access(
    client_id: str,
    user_id: str,
    current_user: Dict[str, Any] = Depends(require_superuser),
):
    """
    Grant a user access to a client.
    
    Requires admin privileges.
    """
    client_store = get_client_store()
    client = await client_store.get(client_id)
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Verify user exists
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    success = await add_user_client(user_id, client_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to grant access")
    
    return {"message": f"User '{user['username']}' granted access to client '{client.name}'"}


@router.delete("/{client_id}/users/{user_id}", summary="Revoke user access from client")
async def revoke_user_access(
    client_id: str,
    user_id: str,
    current_user: Dict[str, Any] = Depends(require_superuser),
):
    """
    Revoke a user's access to a client.
    
    Requires admin privileges.
    """
    client_store = get_client_store()
    client = await client_store.get(client_id)
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Verify user exists
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    success = await remove_user_client(user_id, client_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to revoke access")
    
    return {"message": f"User '{user['username']}' access to client '{client.name}' revoked"}
