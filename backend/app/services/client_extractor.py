"""
Client extraction service using LLM to detect client mentions in chat.
"""
import json
import re
from typing import Optional, Tuple, List

from app.clients.lmstudio import get_lmstudio_client
from app.models.client import Client, get_client_store


EXTRACTION_PROMPT = """You are a client name extraction assistant. Your task is to identify if the user is referring to a specific client/person/entity in their message.

Known clients in the system:
{client_list}

User message: "{message}"

Instructions:
1. If the user mentions a client name (exact or similar), return the client_id
2. If the user says "this client" or "the client" without a name, return "context_needed"
3. If no client is mentioned, return "none"

Respond ONLY with a JSON object:
{{"client_id": "<id or 'context_needed' or 'none'>", "confidence": <0.0-1.0>, "extracted_name": "<name found or null>"}}
"""

CONTEXT_EXTRACTION_PROMPT = """Based on the conversation history, identify which client is being discussed.

Conversation:
{conversation}

Known clients:
{client_list}

If you can identify the client from context, respond with:
{{"client_id": "<id>", "confidence": <0.0-1.0>}}

If unclear, respond with:
{{"client_id": "none", "confidence": 0.0}}
"""


class ClientExtractor:
    """Extracts client references from chat messages using LLM."""
    
    def __init__(self):
        self.client_store = get_client_store()
    
    def _get_client_list_str(self) -> str:
        """Get formatted list of known clients."""
        clients = self.client_store.list_all()
        if not clients:
            return "No clients registered yet."
        
        lines = []
        for c in clients:
            aliases = f" (aliases: {', '.join(c.aliases)})" if c.aliases else ""
            lines.append(f"- ID: {c.id}, Name: {c.name}{aliases}")
        return "\n".join(lines)
    
    def _try_direct_match(self, message: str) -> Optional[Client]:
        """Try to match client name directly without LLM."""
        message_lower = message.lower()
        
        for client in self.client_store.list_all():
            # Check exact name match
            if client.name.lower() in message_lower:
                return client
            
            # Check aliases
            for alias in client.aliases:
                if alias.lower() in message_lower:
                    return client
        
        return None
    
    async def extract_client_from_message(
        self, 
        message: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> Tuple[Optional[Client], float, Optional[str]]:
        """
        Extract client reference from a message.
        
        Returns:
            Tuple of (Client or None, confidence score, extracted_name)
        """
        # First try direct matching (fast path)
        direct_match = self._try_direct_match(message)
        if direct_match:
            return direct_match, 1.0, direct_match.name
        
        # If no clients exist, can't extract
        clients = self.client_store.list_all()
        if not clients:
            return None, 0.0, None
        
        # Use LLM for fuzzy matching
        prompt = EXTRACTION_PROMPT.format(
            client_list=self._get_client_list_str(),
            message=message,
        )
        
        try:
            llm = get_lmstudio_client()
            response = await llm.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )
            
            # Parse JSON response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                data = json.loads(json_match.group())
                client_id = data.get("client_id", "none")
                confidence = float(data.get("confidence", 0.0))
                extracted_name = data.get("extracted_name")
                
                if client_id not in ("none", "context_needed", None):
                    client = self.client_store.get(client_id)
                    if client:
                        return client, confidence, extracted_name
                
                # Try context from conversation history
                if client_id == "context_needed" and conversation_history:
                    return await self._extract_from_context(conversation_history)
        
        except Exception as e:
            print(f"Client extraction error: {e}")
        
        return None, 0.0, None
    
    async def _extract_from_context(
        self,
        conversation_history: List[dict],
    ) -> Tuple[Optional[Client], float, Optional[str]]:
        """Extract client from conversation context."""
        # Format conversation
        conv_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history[-10:]  # Last 10 messages
        ])
        
        prompt = CONTEXT_EXTRACTION_PROMPT.format(
            conversation=conv_text,
            client_list=self._get_client_list_str(),
        )
        
        try:
            llm = get_lmstudio_client()
            response = await llm.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            json_match = re.search(r'\{[^}]+\}', content)
            
            if json_match:
                data = json.loads(json_match.group())
                client_id = data.get("client_id", "none")
                confidence = float(data.get("confidence", 0.0))
                
                if client_id != "none":
                    client = self.client_store.get(client_id)
                    if client:
                        return client, confidence, client.name
        
        except Exception as e:
            print(f"Context extraction error: {e}")
        
        return None, 0.0, None
    
    def create_client_from_name(self, name: str) -> Client:
        """Create a new client with the given name."""
        from app.models.client import ClientCreate
        return self.client_store.create(ClientCreate(name=name))

    async def extract_client(self, message: str, conversation_history: Optional[List[dict]] = None) -> Optional[dict]:
        """
        Extract client reference from message and return as dict.
        
        Returns dict with keys: client_id, client_name, confidence, context
        or None if no client detected.
        """
        client, confidence, extracted_name = await self.extract_client_from_message(
            message, conversation_history
        )
        
        if client is None:
            return None
        
        return {
            "client_id": client.id,
            "client_name": client.name,
            "confidence": confidence,
            "context": extracted_name,
        }


# Singleton
_extractor: Optional[ClientExtractor] = None


def get_client_extractor() -> ClientExtractor:
    """Get singleton client extractor."""
    global _extractor
    if _extractor is None:
        _extractor = ClientExtractor()
    return _extractor
