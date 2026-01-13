"""
Long-Term Memory with Episodic Memory Extraction.

Enhanced for small model RAG optimization:
- Running summaries for older conversation history
- Episodic memory extraction for important decisions/facts
- Structured memory storage for efficient retrieval
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.clients.lmstudio import LMStudioClient, get_lmstudio_client
from app.config import settings
from app.models.schemas import ChatMessage, RetrievalHit
from app.rag.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


SUMMARY_SYSTEM_PROMPT = (
    "You are a concise assistant that summarizes conversations for long-term recall. "
    "Capture key facts, user preferences, and unresolved items in bullet form."
)

RUNNING_SUMMARY_PROMPT = (
    "Create a very brief summary (2-3 sentences max) of the following conversation "
    "that captures the essential context. Focus on: what was discussed, any decisions made, "
    "and what the user is trying to accomplish."
)

EPISODIC_EXTRACTION_PROMPT = """Extract important episodic memories from this conversation.
Focus on:
- User preferences and choices (e.g., "User prefers Python over JavaScript")
- Environment/context facts (e.g., "User's system uses PostgreSQL")
- Decisions made (e.g., "Decided to use option B")
- Constraints mentioned (e.g., "Budget is $1000")

Output format (one per line):
- [memory 1]
- [memory 2]

If no important episodic memories, respond with: "None"

Conversation:
{transcript}
"""


@dataclass
class EpisodicMemory:
    """A single episodic memory (decision, preference, fact)."""
    content: str
    memory_type: str  # "decision", "preference", "fact", "constraint"
    conversation_id: str
    timestamp: int


class EpisodicMemoryExtractor:
    """
    Extracts episodic memories from conversations.
    
    Episodic memories are important facts that should persist:
    - User decisions ("chose option B")
    - Preferences ("prefers SQL output")
    - Environment facts ("uses Postgres")
    - Constraints ("deadline is Friday")
    """
    
    # Patterns that indicate episodic content
    DECISION_PATTERNS = [
        r"\b(chose|choose|selected|select|decided|decide|picked|pick|went with|go with)\b",
        r"\b(prefer|want|would like|need)\b",
        r"\b(let's go with|let's use|we'll use)\b",
    ]
    
    FACT_PATTERNS = [
        r"\b(using|use|runs on|running|environment|system|database|framework)\b",
        r"\b(version|setup|configuration|config)\b",
    ]
    
    CONSTRAINT_PATTERNS = [
        r"\b(budget|deadline|limit|maximum|minimum|constraint|requirement)\b",
        r"\b(must|should|cannot|can't|won't|don't)\b",
        r"\b(\$\d+|\d+\s*(days?|hours?|minutes?|weeks?))\b",
    ]
    
    def __init__(self, lm_client: Optional[LMStudioClient] = None) -> None:
        self.lm_client = lm_client
        self._decision_patterns = [re.compile(p, re.IGNORECASE) for p in self.DECISION_PATTERNS]
        self._fact_patterns = [re.compile(p, re.IGNORECASE) for p in self.FACT_PATTERNS]
        self._constraint_patterns = [re.compile(p, re.IGNORECASE) for p in self.CONSTRAINT_PATTERNS]
    
    async def extract(
        self,
        messages: List[ChatMessage],
        conversation_id: str,
        use_llm: bool = True,
    ) -> List[EpisodicMemory]:
        """
        Extract episodic memories from conversation messages.
        
        Uses pattern matching for quick extraction, optionally
        refined with LLM for better quality.
        """
        if not messages:
            return []
        
        # Quick pattern-based extraction
        pattern_memories = self._extract_by_patterns(messages, conversation_id)
        
        if not use_llm or not self.lm_client:
            return pattern_memories
        
        # LLM-based extraction for better quality
        try:
            llm_memories = await self._extract_with_llm(messages, conversation_id)
            # Combine and deduplicate
            return self._merge_memories(pattern_memories, llm_memories)
        except Exception as e:
            logger.warning(f"LLM episodic extraction failed: {e}")
            return pattern_memories
    
    def _extract_by_patterns(
        self,
        messages: List[ChatMessage],
        conversation_id: str,
    ) -> List[EpisodicMemory]:
        """Extract memories using regex patterns."""
        memories = []
        timestamp = int(time.time())
        
        for msg in messages:
            if msg.role != "user":
                continue
            
            content = msg.content
            
            # Check for decision patterns
            for pattern in self._decision_patterns:
                if pattern.search(content):
                    # Extract the sentence containing the pattern
                    sentences = re.split(r'[.!?]', content)
                    for sentence in sentences:
                        if pattern.search(sentence) and len(sentence.strip()) > 10:
                            memories.append(EpisodicMemory(
                                content=sentence.strip(),
                                memory_type="decision",
                                conversation_id=conversation_id,
                                timestamp=timestamp,
                            ))
                            break
                    break
            
            # Check for constraint patterns
            for pattern in self._constraint_patterns:
                if pattern.search(content):
                    sentences = re.split(r'[.!?]', content)
                    for sentence in sentences:
                        if pattern.search(sentence) and len(sentence.strip()) > 10:
                            memories.append(EpisodicMemory(
                                content=sentence.strip(),
                                memory_type="constraint",
                                conversation_id=conversation_id,
                                timestamp=timestamp,
                            ))
                            break
                    break
        
        # Deduplicate by content similarity
        unique_memories = []
        seen_contents = set()
        for mem in memories:
            content_key = mem.content.lower()[:50]
            if content_key not in seen_contents:
                unique_memories.append(mem)
                seen_contents.add(content_key)
        
        return unique_memories[:5]  # Limit to top 5
    
    async def _extract_with_llm(
        self,
        messages: List[ChatMessage],
        conversation_id: str,
    ) -> List[EpisodicMemory]:
        """Extract memories using LLM."""
        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages[-10:])  # Last 10 messages
        
        prompt = EPISODIC_EXTRACTION_PROMPT.format(transcript=transcript)
        prompt_messages = [{"role": "user", "content": prompt}]
        
        response = await self.lm_client.chat(prompt_messages, stream=False)
        
        # Parse response
        memories = []
        timestamp = int(time.time())
        
        if "none" in response.lower():
            return []
        
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                content = line[2:].strip()
                if len(content) > 10:
                    # Classify memory type
                    memory_type = self._classify_memory(content)
                    memories.append(EpisodicMemory(
                        content=content,
                        memory_type=memory_type,
                        conversation_id=conversation_id,
                        timestamp=timestamp,
                    ))
        
        return memories[:5]
    
    def _classify_memory(self, content: str) -> str:
        """Classify memory type based on content."""
        content_lower = content.lower()
        
        if any(w in content_lower for w in ["chose", "decided", "selected", "picked"]):
            return "decision"
        if any(w in content_lower for w in ["prefer", "want", "like"]):
            return "preference"
        if any(w in content_lower for w in ["budget", "deadline", "limit", "must", "cannot"]):
            return "constraint"
        
        return "fact"
    
    def _merge_memories(
        self,
        pattern_memories: List[EpisodicMemory],
        llm_memories: List[EpisodicMemory],
    ) -> List[EpisodicMemory]:
        """Merge and deduplicate memories from both sources."""
        all_memories = llm_memories + pattern_memories  # Prefer LLM
        
        unique = []
        seen = set()
        
        for mem in all_memories:
            key = mem.content.lower()[:30]
            if key not in seen:
                unique.append(mem)
                seen.add(key)
        
        return unique[:5]


class LongTermMemory:
    """
    Long-term memory with episodic memory support.
    
    Enhanced for small model RAG:
    - summarize_and_store(): Full conversation summary
    - generate_running_summary(): Brief summary for context
    - extract_episodic_memories(): Important decisions/facts
    """
    
    def __init__(self, store: VectorStore, lm_client: LMStudioClient) -> None:
        self.store = store
        self.lm_client = lm_client
        self.episodic_extractor = EpisodicMemoryExtractor(lm_client)

    async def summarize_and_store(
        self, 
        conversation_id: str, 
        messages: List[ChatMessage],
    ) -> RetrievalHit | None:
        """Create and store a full conversation summary."""
        if not messages:
            return None

        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages)
        prompt_messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this conversation:\n{transcript}"},
        ]
        summary = await self.lm_client.chat(prompt_messages, stream=False)
        doc_id = f"{conversation_id}-{int(time.time())}"
        metadata = {"conversation_id": conversation_id, "type": "summary"}
        self.store.add_memories(contents=[summary], ids=[doc_id], metadatas=[metadata])
        return RetrievalHit(id=doc_id, content=summary, score=0.0, metadata=metadata)

    async def generate_running_summary(
        self,
        messages: List[ChatMessage],
        max_length: int = 200,
    ) -> str:
        """
        Generate a brief running summary for older conversation history.
        
        This is used for the sliding window approach - older messages
        get compressed into a short summary.
        """
        if not messages:
            return ""
        
        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages[-10:])
        
        prompt_messages = [
            {"role": "user", "content": f"{RUNNING_SUMMARY_PROMPT}\n\n{transcript}"},
        ]
        
        try:
            summary = await self.lm_client.chat(prompt_messages, stream=False)
            # Ensure it's not too long
            if len(summary) > max_length:
                summary = summary[:max_length].rsplit(' ', 1)[0] + "..."
            return summary
        except Exception as e:
            logger.error(f"Failed to generate running summary: {e}")
            # Fallback: simple truncation of last message
            if messages:
                last_user = next((m for m in reversed(messages) if m.role == "user"), None)
                if last_user:
                    return f"Previously discussed: {last_user.content[:100]}..."
            return ""

    async def extract_and_store_episodics(
        self,
        conversation_id: str,
        messages: List[ChatMessage],
    ) -> List[EpisodicMemory]:
        """
        Extract and store episodic memories from conversation.
        
        Returns the extracted memories.
        """
        if not settings.session.episodic_memory_enabled:
            return []
        
        memories = await self.episodic_extractor.extract(
            messages=messages,
            conversation_id=conversation_id,
            use_llm=True,
        )
        
        # Store in vector DB for retrieval
        if memories:
            contents = [m.content for m in memories]
            ids = [f"episodic-{conversation_id}-{i}-{int(time.time())}" for i in range(len(memories))]
            metadatas = [
                {
                    "conversation_id": conversation_id,
                    "type": "episodic",
                    "memory_type": m.memory_type,
                    "timestamp": m.timestamp,
                }
                for m in memories
            ]
            self.store.add_memories(contents=contents, ids=ids, metadatas=metadatas)
            logger.debug(f"Stored {len(memories)} episodic memories for {conversation_id}")
        
        return memories

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalHit]:
        """Retrieve relevant memories for a query."""
        return self.store.query(query=query, top_k=top_k, collection="memories")
    
    def retrieve_episodics(
        self,
        conversation_id: str,
        top_k: int = 5,
    ) -> List[RetrievalHit]:
        """Retrieve episodic memories for a specific conversation."""
        return self.store.query(
            query="",  # Empty query to get by metadata
            top_k=top_k,
            where={"conversation_id": conversation_id, "type": "episodic"},
            collection="memories",
        )


def get_long_term_memory() -> LongTermMemory:
    """Get singleton long-term memory instance."""
    return LongTermMemory(store=get_vector_store(), lm_client=get_lmstudio_client())

