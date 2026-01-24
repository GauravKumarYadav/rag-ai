"""
Tool Framework for Agentic RAG.

Provides a registry of tools that agents can use to augment
their capabilities beyond document retrieval.

Available Tools:
- Calculator: Mathematical calculations
- DateTime: Date and time operations
- SQLQuery: Client-scoped database queries

Tool Categories:
- Stateless tools: calculator, datetime (no client scope needed)
- Client-scoped tools: sql_query (require client_id for isolation)
"""

from app.tools.base import BaseTool, ToolResult
from app.tools.registry import ToolRegistry, get_tool_registry

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "get_tool_registry",
]
