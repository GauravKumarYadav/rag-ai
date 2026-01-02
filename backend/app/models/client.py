"""Client management models."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
import uuid
import json
import os
from pathlib import Path

from app.config import settings


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
        from difflib import SequenceMatcher
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


class ClientStore:
    """
    Simple file-based client storage.
    In production, use a proper database.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or settings.chroma_db_path) / "clients.json"
        self._clients: dict[str, Client] = {}
        self._load()
    
    def _load(self) -> None:
        """Load clients from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    for client_data in data.get("clients", []):
                        client = Client(**client_data)
                        self._clients[client.id] = client
            except Exception as e:
                print(f"Warning: Could not load clients: {e}")
    
    def _save(self) -> None:
        """Save clients to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "clients": [c.model_dump(mode="json") for c in self._clients.values()]
            }, f, indent=2, default=str)
    
    def create(self, data: ClientCreate) -> Client:
        """Create a new client."""
        client = Client(
            name=data.name,
            aliases=data.aliases,
            metadata=data.metadata,
        )
        self._clients[client.id] = client
        self._save()
        return client
    
    def get(self, client_id: str) -> Optional[Client]:
        """Get client by ID."""
        return self._clients.get(client_id)
    
    def get_by_name(self, name: str, fuzzy: bool = True) -> Optional[Client]:
        """Get client by name (exact or fuzzy match)."""
        # First try exact match
        for client in self._clients.values():
            if client.matches_name(name):
                return client
        
        # Then try fuzzy match
        if fuzzy:
            for client in self._clients.values():
                if client.fuzzy_matches(name):
                    return client
        
        return None
    
    def list_all(self) -> List[Client]:
        """List all clients."""
        return list(self._clients.values())
    
    def update(self, client_id: str, data: ClientUpdate) -> Optional[Client]:
        """Update a client."""
        client = self._clients.get(client_id)
        if not client:
            return None
        
        if data.name is not None:
            client.name = data.name
        if data.aliases is not None:
            client.aliases = data.aliases
        if data.metadata is not None:
            client.metadata = data.metadata
        client.updated_at = datetime.utcnow()
        
        self._clients[client_id] = client
        self._save()
        return client
    
    def delete(self, client_id: str) -> bool:
        """Delete a client."""
        if client_id in self._clients:
            del self._clients[client_id]
            self._save()
            return True
        return False
    
    def search(self, query: str) -> List[Client]:
        """Search clients by name/alias."""
        results = []
        query_lower = query.lower().strip()
        
        for client in self._clients.values():
            # Check if query is substring of name
            if query_lower in client.name.lower():
                results.append(client)
                continue
            
            # Check aliases
            for alias in client.aliases:
                if query_lower in alias.lower():
                    results.append(client)
                    break
        
        return results


# Singleton instance
_client_store: Optional[ClientStore] = None


def get_client_store() -> ClientStore:
    """Get the singleton client store instance."""
    global _client_store
    if _client_store is None:
        _client_store = ClientStore()
    return _client_store
