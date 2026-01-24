"""
Tool Agent for executing tools (calculator, datetime, etc.).

This agent is responsible for:
1. Executing the detected tool
2. Returning the tool result
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from langsmith import traceable

from app.agents.state import AgentState

logger = logging.getLogger(__name__)


class CalculatorTool:
    """
    Calculator tool for mathematical expressions.
    """
    
    @staticmethod
    def execute(expression: str) -> str:
        """
        Execute a mathematical expression.
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            Result as string or error message
        """
        try:
            # Clean up expression
            expression = expression.strip()
            
            # Replace common patterns
            expression = expression.replace('^', '**')  # Power
            expression = expression.replace('×', '*')
            expression = expression.replace('÷', '/')
            
            # Only allow safe characters
            allowed = set('0123456789.+-*/() ')
            if not all(c in allowed for c in expression):
                return "Invalid expression: contains unsupported characters"
            
            # Evaluate safely
            result = eval(expression, {"__builtins__": {}}, {})
            
            # Format result
            if isinstance(result, float):
                # Round to reasonable precision
                if result == int(result):
                    return str(int(result))
                return f"{result:.6g}"
            
            return str(result)
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except SyntaxError:
            return "Error: Invalid expression syntax"
        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return f"Error: {str(e)}"


class DateTimeTool:
    """
    DateTime tool for date and time information.
    """
    
    @staticmethod
    def execute(operation: str = "now") -> str:
        """
        Execute a datetime operation.
        
        Args:
            operation: Operation to perform (now, date, time)
            
        Returns:
            Formatted datetime string
        """
        try:
            now = datetime.now()
            
            if operation == "date":
                return now.strftime("%A, %B %d, %Y")
            elif operation == "time":
                return now.strftime("%I:%M %p")
            else:  # "now" or default
                return now.strftime("%A, %B %d, %Y at %I:%M %p")
                
        except Exception as e:
            logger.error(f"DateTime error: {e}")
            return f"Error: {str(e)}"


class ToolAgent:
    """
    Agent responsible for tool execution.
    """
    
    def __init__(self) -> None:
        self.tools = {
            "calculator": CalculatorTool(),
            "datetime": DateTimeTool(),
        }
    
    @traceable(name="tool_agent.process")
    async def process(self, state: AgentState) -> AgentState:
        """
        Execute the detected tool.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with tool result
        """
        tool_name = state.get("tool_name")
        tool_params = state.get("tool_params", {})
        
        if not tool_name:
            return state
        
        result = await self._execute_tool(tool_name, tool_params)
        
        return {
            **state,
            "tool_result": result,
        }
    
    async def _execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> str:
        """
        Execute a specific tool.
        
        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            
        Returns:
            Tool result as string
        """
        if tool_name == "calculator":
            expression = params.get("expression", "")
            if not expression:
                return "No expression provided"
            return CalculatorTool.execute(expression)
        
        elif tool_name == "datetime":
            operation = params.get("operation", "now")
            return DateTimeTool.execute(operation)
        
        else:
            logger.warning(f"Unknown tool: {tool_name}")
            return f"Unknown tool: {tool_name}"
    
    def list_tools(self) -> list:
        """
        List available tools.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())


# Singleton instance
_tool_agent: Optional[ToolAgent] = None


def get_tool_agent() -> ToolAgent:
    """Get or create the ToolAgent singleton."""
    global _tool_agent
    if _tool_agent is None:
        _tool_agent = ToolAgent()
    return _tool_agent
