"""
Synthesis Agent for response generation.

This agent is responsible for:
1. Building the prompt with context
2. Generating the response
3. Formatting with sources
"""

import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from app.config import settings
from app.agents.state import AgentState
from app.models.schemas import RetrievalHit
from app.core.cost_tracker import get_cost_tracker

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base.
{client_context}

## Response Guidelines:
1. Answer based on the provided context when available
2. Cite sources using [Source: filename] format when using document information
3. If the context doesn't contain enough information, say so clearly
4. Be concise but complete
5. If tool results are provided, incorporate them naturally into your response

## Formatting Guidelines (use Markdown):
- Use **bold** for emphasis and key terms
- Use bullet points or numbered lists for multiple items
- Use tables when comparing data or showing structured information:
  | Column 1 | Column 2 |
  |----------|----------|
  | Data     | Data     |
- Use code blocks with language tags for any code, commands, or technical content:
  ```python
  code here
  ```
- Use > blockquotes for important notes or excerpts
- Use headings (##, ###) to organize longer responses

When no relevant context is found, respond based on your general knowledge but clarify that it's not from the documents."""

CLIENT_CONTEXT_TEMPLATE = """
Current context: You are answering questions for client "{client_name}".
All retrieved documents below are from this client's knowledge base (and any shared global documents).
When the user asks about "my documents" or "available documents", they are referring to {client_name}'s documents.
"""

CHITCHAT_SYSTEM_PROMPT = """You are a friendly AI assistant. Respond naturally to greetings and casual conversation. Keep responses brief and warm."""


class SynthesisAgent:
    """
    Agent responsible for generating responses.
    """
    
    def __init__(self) -> None:
        self.cost_tracker = get_cost_tracker()
        
        # Resolve provider-specific LLM config
        provider = settings.llm_provider.lower()
        if provider == "groq":
            base_url = settings.groq_base_url
            model = settings.groq_model
            api_key = settings.groq_api_key
        elif provider == "openai":
            base_url = settings.openai_base_url
            model = settings.openai_model
            api_key = settings.openai_api_key
        elif provider == "custom":
            base_url = settings.custom_base_url
            model = settings.custom_model
            api_key = settings.custom_api_key
        else:  # default to lmstudio/ollama-compatible endpoint
            base_url = settings.lmstudio_base_url
            model = settings.lmstudio_model
            api_key = "lmstudio"
        
        self.llm = ChatOpenAI(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=settings.llm.timeout,
            callbacks=[self.cost_tracker],
        )
    
    @traceable(name="synthesis_agent.process")
    async def process(self, state: AgentState) -> AgentState:
        """
        Generate a response based on the current state.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with generated response
        """
        intent = state.get("intent", "question")
        message = state.get("message", "")
        client_name = state.get("client_name", "")
        client_id = state.get("client_id", "global")
        
        # Handle chitchat separately
        if intent == "chitchat":
            response = await self._generate_chitchat_response(message)
            return {**state, "response": response}
        
        # Build context for response generation
        context = self._build_context(state)
        tool_context = self._build_tool_context(state)
        conversation_context = self._build_conversation_context(state)
        
        # Generate response with client context
        response = await self._generate_response(
            message=message,
            context=context,
            tool_context=tool_context,
            conversation_context=conversation_context,
            client_name=client_name,
            client_id=client_id,
        )
        
        return {**state, "response": response}
    
    @traceable(name="synthesis_agent.generate_response")
    async def _generate_response(
        self,
        message: str,
        context: str,
        tool_context: str,
        conversation_context: str,
        client_name: str = "",
        client_id: str = "global",
    ) -> str:
        """
        Generate the main response.
        
        Args:
            message: User's message
            context: Document context
            tool_context: Tool results context
            conversation_context: Conversation summary/history
            client_name: Client display name for context
            client_id: Client ID
            
        Returns:
            Generated response text
        """
        # Build client context for system prompt
        if client_id and client_id != "global" and client_name:
            client_context = CLIENT_CONTEXT_TEMPLATE.format(client_name=client_name)
        else:
            client_context = ""
        
        system_prompt = SYSTEM_PROMPT.format(client_context=client_context)
        
        # Build the user prompt
        prompt_parts = []
        
        if conversation_context:
            prompt_parts.append(f"Conversation context:\n{conversation_context}")
        
        if context:
            # Add client info header to document context
            if client_name and client_id != "global":
                prompt_parts.append(f"Documents from {client_name}'s knowledge base:\n{context}")
            else:
                prompt_parts.append(f"Document context:\n{context}")
        else:
            if client_name and client_id != "global":
                prompt_parts.append(f"No documents found in {client_name}'s knowledge base.")
            else:
                prompt_parts.append("No relevant documents found.")
        
        if tool_context:
            prompt_parts.append(f"Tool results:\n{tool_context}")
        
        prompt_parts.append(f"User question: {message}")
        
        user_prompt = "\n\n".join(prompt_parts)
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            
            return response.content or "I'm sorry, I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I encountered an error while generating a response. Please try again."
    
    async def _generate_chitchat_response(self, message: str) -> str:
        """
        Generate a chitchat response.
        
        Args:
            message: User's chitchat message
            
        Returns:
            Friendly response
        """
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ])
            
            return response.content or "Hello! How can I help you today?"
            
        except Exception as e:
            logger.error(f"Chitchat response generation failed: {e}")
            return "Hello! How can I help you?"
    
    def _build_context(self, state: AgentState) -> str:
        """
        Build document context from retrieved chunks.
        
        Args:
            state: Current agent state
            
        Returns:
            Formatted context string
        """
        retrieved = state.get("retrieved_chunks", [])
        if not retrieved:
            return ""
        
        context_parts = []
        for i, hit in enumerate(retrieved[:6], 1):  # Limit to 6 chunks
            source = hit.metadata.get("source", hit.metadata.get("source_filename", f"chunk-{i}"))
            
            # Truncate long content
            content = hit.content
            if len(content) > 600:
                content = content[:600] + "..."
            
            context_parts.append(f"[Source: {source}]\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_tool_context(self, state: AgentState) -> str:
        """
        Build tool results context.
        
        Args:
            state: Current agent state
            
        Returns:
            Formatted tool results string
        """
        tool_result = state.get("tool_result")
        tool_name = state.get("tool_name")
        
        if not tool_result:
            return ""
        
        return f"{tool_name}: {tool_result}"
    
    def _build_conversation_context(self, state: AgentState) -> str:
        """
        Build conversation context from summary and recent messages.
        
        Args:
            state: Current agent state
            
        Returns:
            Formatted conversation context
        """
        summary = state.get("conversation_summary", "")
        recent = state.get("recent_messages", [])
        
        parts = []
        
        if summary:
            parts.append(f"Summary: {summary}")
        
        if recent:
            recent_text = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}"
                for msg in recent[-3:]  # Last 3 messages
            ])
            parts.append(f"Recent:\n{recent_text}")
        
        return "\n\n".join(parts) if parts else ""


# Singleton instance
_synthesis_agent: Optional[SynthesisAgent] = None


def get_synthesis_agent() -> SynthesisAgent:
    """Get or create the SynthesisAgent singleton."""
    global _synthesis_agent
    if _synthesis_agent is None:
        _synthesis_agent = SynthesisAgent()
    return _synthesis_agent
