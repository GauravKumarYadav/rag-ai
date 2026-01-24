"""
Base Tool Interface.

All tools must inherit from BaseTool and implement the execute method.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str
    description: str
    type: str = "string"  # string, number, boolean, array, object
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[str]] = None  # For constrained choices


class ToolResult(BaseModel):
    """Result returned by a tool execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def ok(cls, result: Any, **metadata) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, result=result, metadata=metadata)
    
    @classmethod
    def fail(cls, error: str, **metadata) -> "ToolResult":
        """Create a failed result."""
        return cls(success=False, error=error, metadata=metadata)
    
    def __str__(self) -> str:
        if self.success:
            return str(self.result)
        return f"Error: {self.error}"


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Tools extend agent capabilities by providing specific functionalities
    like calculations, date operations, or database queries.
    
    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description for LLM to understand when to use
        parameters: List of parameter definitions
        requires_client_scope: If True, client_id must be provided for isolation
    """
    
    name: str = "base_tool"
    description: str = "Base tool description"
    parameters: List[ToolParameter] = []
    requires_client_scope: bool = False
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the tool schema for LLM function calling.
        
        Returns a schema compatible with OpenAI function calling format.
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    
    def get_description_for_prompt(self) -> str:
        """
        Get a formatted description for inclusion in prompts.
        """
        param_strs = []
        for p in self.parameters:
            req = "(required)" if p.required else "(optional)"
            param_strs.append(f"  - {p.name}: {p.description} {req}")
        
        params_text = "\n".join(param_strs) if param_strs else "  (no parameters)"
        
        return f"- {self.name}: {self.description}\n{params_text}"
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with success/failure and result data
        """
        pass
    
    def validate_params(self, **kwargs) -> Optional[str]:
        """
        Validate that required parameters are provided.
        
        Returns:
            Error message if validation fails, None otherwise
        """
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return f"Missing required parameter: {param.name}"
            
            if param.enum and param.name in kwargs:
                if kwargs[param.name] not in param.enum:
                    return f"Invalid value for {param.name}. Must be one of: {param.enum}"
        
        return None
