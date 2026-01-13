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
from app.rag.vector_store import get_client_vector_store, clear_client_vector_store_cache
from app.dependencies import get_current_user


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


# =============================================================================
# USER-CLIENT ASSIGNMENT MODELS
# =============================================================================

class UserClientResponse(BaseModel):
    """Response model for user-client assignment."""
    user_id: str
    client_id: str
    message: str


class ClientUserResponse(BaseModel):
    """Response model for user info in client context."""
    id: str
    username: str
    email: Optional[str]
    is_superuser: bool


class UserClientListResponse(BaseModel):
    """Response for listing users of a client."""
    client_id: str
    client_name: str
    users: List[ClientUserResponse]
    total: int


# =============================================================================
# STATIC ROUTES (must come before parameterized routes)
# =============================================================================

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


@router.get("/my/assigned", summary="List my assigned clients")
async def list_my_clients(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    List all clients the current user has access to.
    
    Returns the user's assigned clients, or all clients if user is admin.
    """
    from app.db.mysql import get_user_clients_db
    
    user_id = current_user.get("sub")
    
    if current_user.get("is_superuser"):
        # Admins see all clients
        store = get_client_store()
        all_clients = await store.list_all()
        return {
            "user_id": user_id,
            "is_superuser": True,
            "clients": [
                {
                    "id": c.id,
                    "name": c.name,
                    "aliases": c.aliases,
                }
                for c in all_clients
            ],
            "total": len(all_clients),
        }
    
    # Regular users see only their assigned clients
    clients = await get_user_clients_db(user_id)
    
    return {
        "user_id": user_id,
        "is_superuser": False,
        "clients": [
            {
                "id": c["client_id"],
                "name": c.get("client_name", "Unknown"),
                "role": c.get("role", "viewer"),
            }
            for c in clients
        ],
        "total": len(clients),
    }


@router.get("/user/{user_id}/assigned", summary="List clients for user")
async def list_user_clients(
    user_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    List all clients a specific user has access to.
    
    Regular users can only view their own assignments.
    Admins can view any user's assignments.
    """
    from app.db.mysql import get_user_clients_db
    
    # Check authorization
    if not current_user.get("is_superuser") and current_user.get("sub") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this user's clients")
    
    clients = await get_user_clients_db(user_id)
    
    return {
        "user_id": user_id,
        "clients": [
            {
                "id": c["client_id"],
                "name": c.get("client_name", "Unknown"),
                "role": c.get("role", "viewer"),
            }
            for c in clients
        ],
        "total": len(clients),
    }


# =============================================================================
# PARAMETERIZED ROUTES (must come after static routes)
# =============================================================================

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
    delete_documents: bool = Query(False),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Delete a client.
    
    If delete_documents=True, also deletes all documents associated with the client.
    """
    store = get_client_store()
    client = await store.get(client_id)
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    if delete_documents:
        # Delete client's vector store collections
        client_store = get_client_vector_store(client_id)
        client_store.delete_all()
        clear_client_vector_store_cache(client_id)
    
    await store.delete(client_id)
    
    return {"message": f"Client '{client.name}' deleted", "documents_deleted": delete_documents}


@router.get("/{client_id}/stats", response_model=ClientStatsResponse, summary="Get client stats")
async def get_client_stats(
    client_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get document and memory statistics for a client."""
    client_store = get_client_store()
    client = await client_store.get(client_id)
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    vector_store = get_client_vector_store(client_id)
    stats = vector_store.get_stats()
    
    return ClientStatsResponse(
        client_id=client_id,
        name=client.name,
        document_count=stats["document_count"],
        memory_count=stats["memory_count"],
    )


@router.post("/{client_id}/users/{user_id}", response_model=UserClientResponse, summary="Assign user to client")
async def assign_user_to_client_endpoint(
    client_id: str,
    user_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Assign a user to a client, granting them access to that client's documents.
    
    Requires admin privileges.
    """
    from app.db.mysql import assign_user_to_client, get_user_by_id
    
    # Check if current user is admin
    if not current_user.get("is_superuser"):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    # Verify client exists
    store = get_client_store()
    client = await store.get(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Verify user exists
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Add assignment
    success = await assign_user_to_client(user_id, client_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to assign user to client (may already be assigned)")
    
    return UserClientResponse(
        user_id=user_id,
        client_id=client_id,
        message=f"User '{user['username']}' assigned to client '{client.name}'"
    )


@router.delete("/{client_id}/users/{user_id}", summary="Remove user from client")
async def remove_user_from_client(
    client_id: str,
    user_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Remove a user's access to a client.
    
    Requires admin privileges.
    """
    from app.db.mysql import remove_user_from_client as db_remove_user_from_client
    
    # Check if current user is admin
    if not current_user.get("is_superuser"):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    # Verify client exists
    store = get_client_store()
    client = await store.get(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Remove assignment
    success = await db_remove_user_from_client(user_id, client_id)
    if not success:
        raise HTTPException(status_code=404, detail="User-client assignment not found")
    
    return {"message": f"User removed from client '{client.name}'"}


@router.get("/{client_id}/users", response_model=UserClientListResponse, summary="List users for client")
async def list_client_users(
    client_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    List all users who have access to a specific client.
    
    Requires admin privileges.
    """
    from app.db.mysql import get_client_users_db
    
    # Check if current user is admin
    if not current_user.get("is_superuser"):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    # Verify client exists
    store = get_client_store()
    client = await store.get(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    users = await get_client_users_db(client_id)
    
    return UserClientListResponse(
        client_id=client_id,
        client_name=client.name,
        users=[
            ClientUserResponse(
                id=u["id"],
                username=u["username"],
                email=u.get("email"),
                is_superuser=u.get("is_superuser", False),
            )
            for u in users
        ],
        total=len(users),
    )
