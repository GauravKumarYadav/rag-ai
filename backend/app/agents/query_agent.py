"""
Query Agent for intent classification and query rewriting.

This agent is responsible for:
1. Classifying user intent (chitchat, question, follow_up, tool)
2. Detecting if retrieval is needed
3. Rewriting queries for better retrieval
4. Detecting tool usage
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from app.config import settings
from app.agents.state import AgentState

logger = logging.getLogger(__name__)


INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a RAG chatbot.

Classify the user's message into ONE of these intents:
- chitchat: Greetings, small talk, thanks, or non-question messages
- question: Questions that need to search document CONTENT to answer (e.g., policy details, information lookup)
- document_list: Requests to LIST or ENUMERATE available documents/files (metadata only, not content)
- follow_up: References to previous conversation context
- tool: Requests for calculations, dates, or other tools

IMPORTANT: Distinguish between:
- document_list: User wants to know WHAT documents exist (names, list)
- question: User wants information FROM document content

Return ONLY valid JSON:
{"intent": "question|chitchat|follow_up|tool|document_list", "needs_retrieval": true|false}

Examples:
- "Hello!" → {"intent": "chitchat", "needs_retrieval": false}
- "What is the refund policy?" → {"intent": "question", "needs_retrieval": true}
- "What does my document say?" → {"intent": "question", "needs_retrieval": true}
- "What information is in my Aadhaar card?" → {"intent": "question", "needs_retrieval": true}
- "What documents contain pricing info?" → {"intent": "question", "needs_retrieval": true}
- "Tell me more about that" → {"intent": "follow_up", "needs_retrieval": true}
- "What is 15% of 200?" → {"intent": "tool", "needs_retrieval": false}
- "What documents do I have?" → {"intent": "document_list", "needs_retrieval": false}
- "List my files" → {"intent": "document_list", "needs_retrieval": false}
- "Show all uploaded documents" → {"intent": "document_list", "needs_retrieval": false}
- "What's in my knowledge base?" → {"intent": "document_list", "needs_retrieval": false}
"""

QUERY_REWRITE_PROMPT = """Rewrite the user's query to be more effective for document retrieval.

Guidelines:
- Remove conversational prefixes like "can you tell me", "please help me"
- Make the query standalone (resolve references using conversation context)
- Focus on key terms and concepts
- Keep it concise but complete

Conversation context: {context}

Original query: {query}

Return ONLY the rewritten query, nothing else.
"""


class QueryAgent:
    """
    Agent responsible for query understanding and classification.
    """
    
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            base_url=settings.llm.lmstudio.base_url,
            model=settings.llm.lmstudio.model,
            api_key="lmstudio",
            temperature=0.1,  # Low temperature for classification
            max_tokens=256,
            timeout=settings.llm.timeout,
        )
    
    @traceable(name="query_agent.process")
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the query: classify intent, detect tools, rewrite query.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with intent, tool detection, and rewritten query
        """
        message = state.get("message", "")
        conversation_summary = state.get("conversation_summary", "")
        
        # Step 1: Classify intent via LLM (including document_list detection)
        intent, needs_retrieval = await self._classify_intent(message, conversation_summary)
        
        # Step 2: Detect tools (overrides LLM classification if tool detected)
        tool_name, tool_params = self._detect_tool(message)
        if tool_name:
            intent = "tool"
            needs_retrieval = False
        
        # Step 3: Rewrite query if retrieval needed
        rewritten_query = message
        if needs_retrieval and intent in ("question", "follow_up"):
            rewritten_query = await self._rewrite_query(message, conversation_summary)
        
        return {
            **state,
            "intent": intent,
            "needs_retrieval": needs_retrieval,
            "tool_name": tool_name,
            "tool_params": tool_params,
            "rewritten_query": rewritten_query,
        }
    
    @traceable(name="query_agent.classify_intent")
    async def _classify_intent(self, message: str, conversation_summary: str = "") -> Tuple[str, bool]:
        """
        Classify the user's intent using LLM.
        
        Args:
            message: User's message
            conversation_summary: Optional conversation history context
        
        Returns:
            Tuple of (intent, needs_retrieval)
        """
        intent = "question"
        needs_retrieval = True
        
        try:
            # Build context-aware prompt
            user_content = f"User message: {message}"
            if conversation_summary:
                user_content = f"Conversation context: {conversation_summary[:300]}\n\n{user_content}"
            
            response = await self.llm.ainvoke([
                SystemMessage(content=INTENT_CLASSIFICATION_PROMPT),
                HumanMessage(content=user_content),
            ])
            
            text = response.content or ""
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                data = json.loads(json_match.group())
                intent = str(data.get("intent", intent)).strip().lower()
                needs_retrieval = bool(data.get("needs_retrieval", needs_retrieval))
                
                # Validate intent
                valid_intents = {"chitchat", "question", "follow_up", "tool", "document_list"}
                if intent not in valid_intents:
                    intent = "question"
                    needs_retrieval = True
                    
        except Exception as e:
            logger.debug(f"Intent classification failed: {e}, defaulting to question")
        
        return intent, needs_retrieval
    
    @traceable(name="query_agent.rewrite_query")
    async def _rewrite_query(self, query: str, context: str) -> str:
        """
        Rewrite query for better retrieval.
        
        Args:
            query: Original user query
            context: Conversation context/summary
            
        Returns:
            Rewritten query
        """
        if not context:
            # No context, just clean up the query
            return self._clean_query(query)
        
        try:
            prompt = QUERY_REWRITE_PROMPT.format(
                context=context[:500],  # Limit context length
                query=query,
            )
            
            response = await self.llm.ainvoke([
                HumanMessage(content=prompt),
            ])
            
            rewritten = (response.content or "").strip()
            if rewritten and len(rewritten) > 3:
                return rewritten
                
        except Exception as e:
            logger.debug(f"Query rewrite failed: {e}")
        
        return self._clean_query(query)
    
    def _clean_query(self, query: str) -> str:
        """
        Clean up query by removing conversational prefixes.
        """
        prefixes = [
            "can you tell me",
            "please tell me",
            "i want to know",
            "could you explain",
            "help me understand",
            "what do you think about",
        ]
        
        query_lower = query.lower().strip()
        for prefix in prefixes:
            if query_lower.startswith(prefix):
                query = query[len(prefix):].strip()
                # Remove leading punctuation
                query = query.lstrip("?,. ")
                break
        
        return query
    
    def _detect_tool(self, message: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Detect if the message requires a tool.
        
        Returns:
            Tuple of (tool_name, tool_params) or (None, {})
        """
        message_lower = message.lower()
        
        # Calculator detection
        math_pattern = r'[\d\.\s\+\-\*\/\(\)\^%]+'
        matches = re.findall(math_pattern, message)
        for match in matches:
            if any(op in match for op in ['+', '-', '*', '/', '^', '%']):
                # Clean up the expression
                expression = match.strip()
                if expression:
                    return "calculator", {"expression": expression}
        
        # Percentage calculations
        percent_pattern = r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)'
        percent_match = re.search(percent_pattern, message_lower)
        if percent_match:
            pct, value = percent_match.groups()
            expression = f"{pct} / 100 * {value}"
            return "calculator", {"expression": expression}
        
        # DateTime detection
        datetime_keywords = ["date", "time", "today", "now", "current time", "what day"]
        if any(kw in message_lower for kw in datetime_keywords):
            return "datetime", {"operation": "now"}
        
        return None, {}


# Singleton instance
_query_agent: Optional[QueryAgent] = None


def get_query_agent() -> QueryAgent:
    """Get or create the QueryAgent singleton."""
    global _query_agent
    if _query_agent is None:
        _query_agent = QueryAgent()
    return _query_agent
