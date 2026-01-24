"""
Tool Registry.

Central registry for tool discovery and management.
"""

import logging
from functools import lru_cache
from typing import Dict, List, Optional

from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry of available tools for agent use.
    
    Provides:
    - Tool registration and discovery
    - Tool descriptions for LLM prompts
    - Tool execution with validation
    """
    
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
        """
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")
        
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry.
        
        Args:
            name: Name of tool to remove
            
        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def get_all(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_tool_descriptions(self) -> str:
        """
        Get formatted descriptions of all tools for LLM context.
        
        Returns:
            Formatted string describing all available tools
        """
        if not self._tools:
            return "No tools available."
        
        descriptions = []
        for tool in self._tools.values():
            descriptions.append(tool.get_description_for_prompt())
        
        return "\n\n".join(descriptions)
    
    def get_schemas(self) -> List[Dict]:
        """
        Get schemas for all tools (for function calling).
        
        Returns:
            List of tool schemas in OpenAI function format
        """
        return [tool.get_schema() for tool in self._tools.values()]
    
    async def execute(
        self,
        tool_name: str,
        client_id: Optional[str] = None,
        **params
    ) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of tool to execute
            client_id: Client ID for client-scoped tools
            **params: Tool parameters
            
        Returns:
            ToolResult with execution outcome
        """
        tool = self.get(tool_name)
        
        if tool is None:
            return ToolResult.fail(f"Tool not found: {tool_name}")
        
        # Check client scope requirement
        if tool.requires_client_scope and not client_id:
            return ToolResult.fail(f"Tool {tool_name} requires client_id for isolation")
        
        # Inject client_id if tool requires it
        if tool.requires_client_scope:
            params["client_id"] = client_id
        
        # Validate parameters
        validation_error = tool.validate_params(**params)
        if validation_error:
            return ToolResult.fail(validation_error)
        
        try:
            return await tool.execute(**params)
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return ToolResult.fail(str(e))


# Global registry instance
_registry: Optional[ToolRegistry] = None


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry with default tools registered.
    """
    registry = ToolRegistry()
    
    # Register default tools
    try:
        from app.tools.calculator import CalculatorTool
        registry.register(CalculatorTool())
    except ImportError:
        logger.warning("Calculator tool not available")
    
    try:
        from app.tools.datetime_tool import DateTimeTool
        registry.register(DateTimeTool())
    except ImportError:
        logger.warning("DateTime tool not available")
    
    try:
        from app.tools.sql_query import SQLQueryTool
        registry.register(SQLQueryTool())
    except ImportError:
        logger.warning("SQL Query tool not available")
    
    logger.info(f"Tool registry initialized with {len(registry.list_tools())} tools")
    
    return registry
