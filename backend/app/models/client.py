"""Client management models with MySQL persistence."""
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional, List
from pydantic import BaseModel, Field
import uuid

from app.db.mysql import (
    create_client_db,
    get_client_db,
    get_client_by_name_db,
    list_clients_db,
    update_client_db,
    delete_client_db,
    search_clients_db,
)


class Client(BaseModel):
    """Represents a client entity."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    aliases: List[str] = Field(default_factory=list)  # Alternative names/spellings
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def matches_name(self, query: str) -> bool:
        """Check if query matches client name or aliases (case-insensitive)."""
        query_lower = query.lower().strip()
        if self.name.lower() == query_lower:
            return True
        for alias in self.aliases:
            if alias.lower() == query_lower:
                return True
        return False
    
    def fuzzy_matches(self, query: str, threshold: float = 0.8) -> bool:
        """Check if query fuzzy-matches client name or aliases."""
        query_lower = query.lower().strip()
        
        # Check main name
        ratio = SequenceMatcher(None, self.name.lower(), query_lower).ratio()
        if ratio >= threshold:
            return True
        
        # Check aliases
        for alias in self.aliases:
            ratio = SequenceMatcher(None, alias.lower(), query_lower).ratio()
            if ratio >= threshold:
                return True
        
        return False


class ClientCreate(BaseModel):
    """Request model for creating a client."""
    name: str
    aliases: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ClientUpdate(BaseModel):
    """Request model for updating a client."""
    name: Optional[str] = None
    aliases: Optional[List[str]] = None
    metadata: Optional[dict] = None


def _dict_to_client(data: dict) -> Client:
    """Convert a database dict to a Client model."""
    return Client(
        id=data['id'],
        name=data['name'],
        aliases=data.get('aliases', []),
        metadata=data.get('metadata', {}),
        created_at=data.get('created_at', datetime.utcnow()),
        updated_at=data.get('updated_at', datetime.utcnow()),
    )


class MySQLClientStore:
    """
    MySQL-backed client storage.
    Persists client data across container recreations.
    """
    
    async def create(self, data: ClientCreate) -> Client:
        """Create a new client."""
        client_id = str(uuid.uuid4())
        result = await create_client_db(
            client_id=client_id,
            name=data.name,
            aliases=data.aliases,
            metadata=data.metadata,
        )
        if result:
            return _dict_to_client(result)
        raise ValueError(f"Failed to create client: {data.name}")
    
    async def get(self, client_id: str) -> Optional[Client]:
        """Get client by ID."""
        result = await get_client_db(client_id)
        if result:
            return _dict_to_client(result)
        return None
    
    async def get_by_name(self, name: str, fuzzy: bool = True) -> Optional[Client]:
        """Get client by name (exact or fuzzy match)."""
        # First try exact match
        result = await get_client_by_name_db(name)
        if result:
            return _dict_to_client(result)
        
        # Check aliases with exact match
        all_clients = await list_clients_db()
        for client_data in all_clients:
            client = _dict_to_client(client_data)
            if client.matches_name(name):
                return client
        
        # Then try fuzzy match if enabled
        if fuzzy:
            for client_data in all_clients:
                client = _dict_to_client(client_data)
                if client.fuzzy_matches(name):
                    return client
        
        return None
    
    async def list_all(self) -> List[Client]:
        """List all clients."""
        results = await list_clients_db()
        return [_dict_to_client(r) for r in results]
    
    async def update(self, client_id: str, data: ClientUpdate) -> Optional[Client]:
        """Update a client."""
        result = await update_client_db(
            client_id=client_id,
            name=data.name,
            aliases=data.aliases,
            metadata=data.metadata,
        )
        if result:
            return _dict_to_client(result)
        return None
    
    async def delete(self, client_id: str) -> bool:
        """Delete a client."""
        return await delete_client_db(client_id)
    
    async def search(self, query: str) -> List[Client]:
        """Search clients by name/alias."""
        # Search by name in database
        results = await search_clients_db(query)
        clients = [_dict_to_client(r) for r in results]
        
        # Also check aliases (not stored in DB search)
        all_clients = await list_clients_db()
        query_lower = query.lower().strip()
        
        found_ids = {c.id for c in clients}
        for client_data in all_clients:
            if client_data['id'] in found_ids:
                continue
            for alias in client_data.get('aliases', []):
                if query_lower in alias.lower():
                    clients.append(_dict_to_client(client_data))
                    break
        
        return clients


# Singleton instance
_client_store: Optional[MySQLClientStore] = None


def get_client_store() -> MySQLClientStore:
    """Get the singleton client store instance."""
    global _client_store
    if _client_store is None:
        _client_store = MySQLClientStore()
    return _client_store


# Backward compatibility alias
ClientStore = MySQLClientStore
