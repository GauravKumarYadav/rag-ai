"""
Tool Agent.

Selects and executes appropriate tools based on the query.
Ensures client isolation for tools that require it.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.agents.state import AgentAction, AgentState, ActionType
from app.clients.lmstudio import LMStudioClient
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry, get_tool_registry
from app.services.query_processor import DOCUMENT_LIST_PATTERNS

logger = logging.getLogger(__name__)


TOOL_SELECTION_PROMPT = """You are a tool selection assistant. Analyze the query and decide if a tool should be used.

Query: {query}

Available tools:
{tool_descriptions}

Context already retrieved: {context_summary}

Based on the query, should a tool be used?
- Use calculator for mathematical calculations
- Use datetime for date/time questions or calculations
- Use sql_query only for explicit SQL or count queries (not for listing documents)

Output JSON:
{{
    "use_tool": true/false,
    "tool_name": "calculator|datetime|sql_query" or null,
    "parameters": {{"param1": "value1"}} or {{}},
    "reasoning": "Brief explanation"
}}

If no tool is needed, set use_tool to false.
Output ONLY valid JSON:"""


class ToolAgent(BaseAgent):
    """
    Selects and executes tools based on query analysis.
    
    Features:
    - Automatic tool selection based on query
    - Client-scoped execution for isolated tools
    - Parameter extraction from query
    - Error handling and fallback
    """
    
    name: str = "tool_agent"
    description: str = "Selects and executes appropriate tools"
    
    def __init__(
        self,
        lm_client: LMStudioClient,
        registry: Optional[ToolRegistry] = None,
        allowed_tools: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(lm_client, max_iterations=1, verbose=verbose)
        self.registry = registry or get_tool_registry()
        self.allowed_tools = allowed_tools  # None means all tools allowed
        self._document_list_patterns = [re.compile(p, re.IGNORECASE) for p in DOCUMENT_LIST_PATTERNS]
    
    async def run(self, state: AgentState) -> AgentState:
        """
        Analyze query and execute tools if needed.
        """
        logger.debug(f"[{self.name}] Analyzing query for tool usage")
        
        if self._is_document_list_query(state.query):
            state.add_observation("Document list query - skipping tool selection")
            return state
        
        # Check if plan specifies tools
        tools_to_use = []
        if state.plan and state.plan.tool_names:
            tools_to_use = state.plan.tool_names
        
        if not tools_to_use:
            # Use LLM to decide on tool usage
            tool_decision = await self._decide_tool(state)
            
            if tool_decision.get("use_tool") and tool_decision.get("tool_name"):
                tools_to_use = [tool_decision["tool_name"]]
                
                # Execute the tool
                result = await self._execute_tool(
                    tool_name=tool_decision["tool_name"],
                    params=tool_decision.get("parameters", {}),
                    state=state
                )
                
                if result.success:
                    state.tool_results[tool_decision["tool_name"]] = result.result
                    state.add_observation(f"Tool {tool_decision['tool_name']}: {result.result}")
                else:
                    state.add_observation(f"Tool {tool_decision['tool_name']} failed: {result.error}")
        else:
            # Execute tools specified in plan
            for tool_name in tools_to_use:
                # Extract parameters for this tool from query
                params = await self._extract_params(tool_name, state)
                
                result = await self._execute_tool(tool_name, params, state)
                
                if result.success:
                    state.tool_results[tool_name] = result.result
                    state.add_observation(f"Tool {tool_name}: {result.result}")
                else:
                    state.add_observation(f"Tool {tool_name} failed: {result.error}")
        
        logger.info(f"[{self.name}] Executed {len(state.tool_results)} tools")
        
        return state
    
    async def think(self, state: AgentState) -> str:
        """Analyze if tools are needed."""
        if state.plan and state.plan.tool_names:
            return f"Plan specifies tools: {state.plan.tool_names}"
        
        # Quick heuristic check
        query_lower = state.query.lower()
        
        if any(kw in query_lower for kw in ["calculate", "compute", "sum", "total", "+", "-", "*", "/"]):
            return "Query may need calculator tool"
        
        if any(kw in query_lower for kw in ["date", "time", "today", "yesterday", "ago", "days"]):
            return "Query may need datetime tool"
        
        return "Query may not need tools"
    
    async def act(self, state: AgentState, thought: str) -> AgentAction:
        """Decide on tool action."""
        if "calculator" in thought.lower():
            return AgentAction.use_tool("calculator", {})
        
        if "datetime" in thought.lower():
            return AgentAction.use_tool("datetime", {})
        
        if "sql" in thought.lower():
            return AgentAction.use_tool("sql_query", {})
        
        return AgentAction.stop("No tools needed")
    
    async def observe(self, state: AgentState, action: AgentAction) -> AgentState:
        """Process tool execution results."""
        if state.tool_results:
            tools_used = list(state.tool_results.keys())
            state.add_observation(f"Tools executed: {tools_used}")
        return state
    
    async def _decide_tool(self, state: AgentState) -> Dict[str, Any]:
        """
        Use LLM to decide which tool to use.
        """
        if self._is_document_list_query(state.query):
            return {"use_tool": False, "reasoning": "Document list handled outside tools"}

        # Get tool descriptions
        tool_descriptions = self.registry.get_tool_descriptions()
        
        # Filter by allowed tools if specified
        if self.allowed_tools:
            filtered_tools = [
                t for t in self.registry.get_all()
                if t.name in self.allowed_tools
            ]
            tool_descriptions = "\n\n".join(
                t.get_description_for_prompt() for t in filtered_tools
            )
        
        prompt = TOOL_SELECTION_PROMPT.format(
            query=state.query,
            tool_descriptions=tool_descriptions,
            context_summary=state.get_context_summary()
        )
        
        try:
            response = await self._call_llm(prompt)
            return self._parse_tool_decision(response)
        except Exception as e:
            logger.warning(f"Tool decision failed: {e}")
            return {"use_tool": False}
    
    def _parse_tool_decision(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into tool decision."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "use_tool": data.get("use_tool", False),
                    "tool_name": data.get("tool_name"),
                    "parameters": data.get("parameters", {}),
                    "reasoning": data.get("reasoning", "")
                }
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Failed to parse tool decision: {e}")
        
        return {"use_tool": False}

    def _is_document_list_query(self, query: str) -> bool:
        """Check if the query is asking to list documents."""
        query_lower = query.lower().strip()
        return any(pattern.search(query_lower) for pattern in self._document_list_patterns)
    
    async def _extract_params(self, tool_name: str, state: AgentState) -> Dict[str, Any]:
        """
        Extract tool parameters from query using LLM.
        """
        tool = self.registry.get(tool_name)
        if not tool:
            return {}
        
        # Get parameter schema
        param_desc = "\n".join(
            f"- {p.name}: {p.description}" 
            for p in tool.parameters
        )
        
        prompt = f"""Extract parameters for the {tool_name} tool from this query.

Query: {state.query}

Required parameters:
{param_desc}

Output JSON with parameter values:"""
        
        try:
            response = await self._call_llm(prompt)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.debug(f"Parameter extraction failed: {e}")
        
        # Fallback: try to extract based on tool type
        return self._extract_params_heuristic(tool_name, state.query)
    
    def _extract_params_heuristic(self, tool_name: str, query: str) -> Dict[str, Any]:
        """
        Heuristic parameter extraction without LLM.
        """
        if tool_name == "calculator":
            # Try to find mathematical expression
            # Look for patterns like "2 + 2", "100 * 0.15", etc.
            math_pattern = r'[\d\.\s\+\-\*\/\(\)\^]+'
            matches = re.findall(math_pattern, query)
            for match in matches:
                if any(op in match for op in ['+', '-', '*', '/', '^']):
                    return {"expression": match.strip()}
            return {}
        
        if tool_name == "datetime":
            # Check for common date operations
            query_lower = query.lower()
            if any(kw in query_lower for kw in ["today", "now", "current time", "current date"]):
                return {"operation": "now"}
            if any(kw in query_lower for kw in ["between", "from", "since", "until"]):
                return {}
            return {"operation": "now"}
        
        if tool_name == "sql_query":
            # Check for document count queries
            if "how many" in query.lower() or "count" in query.lower():
                return {"query": "SELECT COUNT(*) as count FROM documents"}
            return {}
        
        return {}
    
    async def _execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        state: AgentState
    ) -> ToolResult:
        """
        Execute a tool with proper client isolation.
        """
        # Check if tool is allowed
        if self.allowed_tools and tool_name not in self.allowed_tools:
            return ToolResult.fail(f"Tool not allowed: {tool_name}")
        
        # Execute through registry (handles client_id injection)
        return await self.registry.execute(
            tool_name=tool_name,
            client_id=state.client_id,
            **params
        )


def get_tool_agent(
    lm_client: Optional[LMStudioClient] = None,
    allowed_tools: Optional[List[str]] = None,
) -> ToolAgent:
    """Factory function to create tool agent."""
    if lm_client is None:
        from app.clients.lmstudio import get_lmstudio_client
        lm_client = get_lmstudio_client()
    
    # Get allowed tools from config if available
    if allowed_tools is None:
        try:
            from app.config import settings
            if hasattr(settings, 'agent') and hasattr(settings.agent, 'allowed_tools'):
                allowed_tools = settings.agent.allowed_tools
        except (ImportError, AttributeError):
            pass
    
    return ToolAgent(
        lm_client=lm_client,
        allowed_tools=allowed_tools,
    )
