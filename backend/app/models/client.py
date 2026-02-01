"""Client management models with Redis persistence."""

import json
import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional, List
from pydantic import BaseModel, Field
import uuid

import redis

from app.config import settings

logger = logging.getLogger(__name__)

# Redis keys
CLIENTS_KEY = "clients:data"
CLIENTS_INDEX_KEY = "clients:index"

_redis_client: Optional[redis.Redis] = None


def _get_redis() -> Optional[redis.Redis]:
    """Get Redis client, initializing if needed."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(settings.redis.url, decode_responses=True)
            _redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed for clients: {e}")
            _redis_client = None
    return _redis_client


class Client(BaseModel):
    """Represents a client entity."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    aliases: List[str] = Field(default_factory=list)
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
        
        ratio = SequenceMatcher(None, self.name.lower(), query_lower).ratio()
        if ratio >= threshold:
            return True
        
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


def _ensure_global_client() -> None:
    """Ensure the global client exists."""
    r = _get_redis()
    if r is None:
        return
    
    if not r.hexists(CLIENTS_KEY, "global"):
        global_client = {
            "id": "global",
            "name": "Global",
            "aliases": ["global", "default", "shared"],
            "metadata": {"description": "Global documents available to all clients"},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        r.hset(CLIENTS_KEY, "global", json.dumps(global_client))
        logger.info("Created global client")


def _dict_to_client(data: dict) -> Client:
    """Convert a dict to a Client model."""
    created_at = data.get('created_at')
    updated_at = data.get('updated_at')
    
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    elif created_at is None:
        created_at = datetime.utcnow()
    
    if isinstance(updated_at, str):
        updated_at = datetime.fromisoformat(updated_at)
    elif updated_at is None:
        updated_at = datetime.utcnow()
    
    return Client(
        id=data['id'],
        name=data['name'],
        aliases=data.get('aliases', []),
        metadata=data.get('metadata', {}),
        created_at=created_at,
        updated_at=updated_at,
    )


class RedisClientStore:
    """
    Redis-backed client storage.
    Persists client data across container recreations.
    """
    
    def __init__(self):
        _ensure_global_client()
    
    async def create(self, data: ClientCreate) -> Client:
        """Create a new client."""
        r = _get_redis()
        if r is None:
            raise RuntimeError("Redis not available")
        
        client_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        client_data = {
            "id": client_id,
            "name": data.name,
            "aliases": data.aliases,
            "metadata": data.metadata,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        
        r.hset(CLIENTS_KEY, client_id, json.dumps(client_data))
        return _dict_to_client(client_data)
    
    async def get(self, client_id: str) -> Optional[Client]:
        """Get client by ID."""
        _ensure_global_client()
        r = _get_redis()
        if r is None:
            if client_id == "global":
                return Client(
                    id="global",
                    name="Global",
                    aliases=["global", "default"],
                    metadata={},
                )
            return None
        
        data = r.hget(CLIENTS_KEY, client_id)
        if data:
            return _dict_to_client(json.loads(data))
        return None
    
    async def get_by_name(self, name: str, fuzzy: bool = True) -> Optional[Client]:
        """Get client by name (exact or fuzzy match)."""
        all_clients = await self.list_all()
        
        # Exact match
        for client in all_clients:
            if client.matches_name(name):
                return client
        
        # Fuzzy match
        if fuzzy:
            for client in all_clients:
                if client.fuzzy_matches(name):
                    return client
        
        return None
    
    async def list_all(self) -> List[Client]:
        """List all clients."""
        _ensure_global_client()
        r = _get_redis()
        if r is None:
            return [Client(id="global", name="Global", aliases=["global"])]
        
        all_data = r.hgetall(CLIENTS_KEY)
        clients = []
        for client_data in all_data.values():
            try:
                clients.append(_dict_to_client(json.loads(client_data)))
            except Exception as e:
                logger.warning(f"Failed to parse client data: {e}")
        
        return clients
    
    async def update(self, client_id: str, data: ClientUpdate) -> Optional[Client]:
        """Update a client."""
        r = _get_redis()
        if r is None:
            return None
        
        existing = r.hget(CLIENTS_KEY, client_id)
        if not existing:
            return None
        
        client_data = json.loads(existing)
        
        if data.name is not None:
            client_data["name"] = data.name
        if data.aliases is not None:
            client_data["aliases"] = data.aliases
        if data.metadata is not None:
            client_data["metadata"] = data.metadata
        
        client_data["updated_at"] = datetime.utcnow().isoformat()
        
        r.hset(CLIENTS_KEY, client_id, json.dumps(client_data))
        return _dict_to_client(client_data)
    
    async def delete(self, client_id: str) -> bool:
        """Delete a client."""
        if client_id == "global":
            return False  # Can't delete global client
        
        r = _get_redis()
        if r is None:
            return False
        
        return r.hdel(CLIENTS_KEY, client_id) > 0
    
    async def search(self, query: str) -> List[Client]:
        """Search clients by name/alias."""
        all_clients = await self.list_all()
        query_lower = query.lower().strip()
        
        results = []
        for client in all_clients:
            if query_lower in client.name.lower():
                results.append(client)
            elif any(query_lower in alias.lower() for alias in client.aliases):
                results.append(client)
        
        return results


# Singleton instance
_client_store: Optional[RedisClientStore] = None


def get_client_store() -> RedisClientStore:
    """Get the singleton client store instance."""
    global _client_store
    if _client_store is None:
        _client_store = RedisClientStore()
    return _client_store


# Backward compatibility alias
ClientStore = RedisClientStore
