"""Client management API endpoints."""
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.models.client import (
    Client,
    ClientCreate,
    ClientUpdate,
    get_client_store,
)
from app.rag.vector_store import get_client_vector_store, clear_client_vector_store_cache


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
async def create_client(data: ClientCreate):
    """Create a new client for document organization."""
    store = get_client_store()
    
    # Check if client with same name exists
    existing = store.get_by_name(data.name, fuzzy=False)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Client with name '{data.name}' already exists (ID: {existing.id})",
        )
    
    client = store.create(data)
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
):
    """List all clients, optionally filtered by search query."""
    store = get_client_store()
    
    if search:
        clients = store.search(search)
    else:
        clients = store.list_all()
    
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


@router.get("/{client_id}", response_model=ClientResponse, summary="Get client by ID")
async def get_client(client_id: str):
    """Get a specific client by ID."""
    store = get_client_store()
    client = store.get(client_id)
    
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
async def update_client(client_id: str, data: ClientUpdate):
    """Update a client's information."""
    store = get_client_store()
    client = store.update(client_id, data)
    
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
async def delete_client(client_id: str, delete_documents: bool = Query(False)):
    """
    Delete a client.
    
    If delete_documents=True, also deletes all documents associated with the client.
    """
    store = get_client_store()
    client = store.get(client_id)
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    if delete_documents:
        # Delete client's vector store collections
        client_store = get_client_vector_store(client_id)
        client_store.delete_all()
        clear_client_vector_store_cache(client_id)
    
    store.delete(client_id)
    
    return {"message": f"Client '{client.name}' deleted", "documents_deleted": delete_documents}


@router.get("/{client_id}/stats", response_model=ClientStatsResponse, summary="Get client stats")
async def get_client_stats(client_id: str):
    """Get document and memory statistics for a client."""
    client_store = get_client_store()
    client = client_store.get(client_id)
    
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


@router.get("/search/by-name", response_model=ClientResponse, summary="Find client by name")
async def find_client_by_name(
    name: str = Query(..., description="Client name to search"),
    fuzzy: bool = Query(True, description="Allow fuzzy matching"),
):
    """Find a client by name (exact or fuzzy match)."""
    store = get_client_store()
    client = store.get_by_name(name, fuzzy=fuzzy)
    
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
