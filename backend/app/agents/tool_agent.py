"""
Tool Agent for executing tools (calculator, datetime, etc.).

This agent is responsible for:
1. Executing the detected tool
2. Returning the tool result
"""

import logging
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langsmith import traceable

from app.agents.state import AgentState

logger = logging.getLogger(__name__)

# Pre-compiled patterns for zero-allocation matching
TOKEN_PATTERN = re.compile(r'\d+\.?\d*|\+|\-|\*|\/|\^|\(|\)|%')
SUSPICIOUS_PATTERNS = ['import', 'eval', 'exec', 'compile', '__', 'lambda', ';']


class SafeCalculator:
    """
    Production-safe calculator with O(n) parsing and minimal memory.
    Uses shunting-yard algorithm for safe infix expression evaluation.
    """
    
    # Operator precedence (higher = evaluated first)
    PRECEDENCE = {'+': 1, '-': 1, '*': 2, '/': 2, '%': 2, '^': 3}
    
    @classmethod
    def execute(cls, expression: str) -> str:
        """
        Execute a mathematical expression safely.
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            Result as string or error message
        """
        try:
            # Fast rejection of suspicious input
            if cls._contains_suspicious_patterns(expression):
                return "Error: Invalid characters in expression"
            
            # Tokenize and validate
            tokens = cls._tokenize(expression)
            if not tokens:
                return "Error: Empty expression"
            
            # Evaluate using shunting-yard algorithm
            result = cls._evaluate_tokens(tokens)
            return cls._format_result(result)
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except (ValueError, SyntaxError) as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.warning(f"Calculator evaluation error: {e}")
            return "Error: Could not evaluate expression"
    
    @classmethod
    def _contains_suspicious_patterns(cls, expr: str) -> bool:
        """Fast check for dangerous patterns."""
        expr_lower = expr.lower()
        return any(dangerous in expr_lower for dangerous in SUSPICIOUS_PATTERNS)
    
    @classmethod
    def _tokenize(cls, expr: str) -> List[str]:
        """Tokenize with bounds checking."""
        if not expr:
            return []
        
        # Normalize unicode operators
        expr = expr.replace('×', '*').replace('÷', '/').replace('−', '-')
        
        tokens = TOKEN_PATTERN.findall(expr)
        
        # Validate token count (prevent memory exhaustion)
        if len(tokens) > 100:
            raise ValueError("Expression too complex")
        
        return tokens
    
    @classmethod
    def _evaluate_tokens(cls, tokens: List[str]) -> float:
        """Shunting-yard algorithm for safe infix evaluation."""
        output: List[float] = []
        operators: List[str] = []
        
        for token in tokens:
            if cls._is_number(token):
                output.append(float(token))
            elif token in cls.PRECEDENCE:
                # Pop operators with higher or equal precedence
                while (operators and 
                       operators[-1] in cls.PRECEDENCE and 
                       cls.PRECEDENCE[operators[-1]] >= cls.PRECEDENCE[token]):
                    cls._apply_operator(output, operators.pop())
                operators.append(token)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                # Pop until matching '('
                while operators and operators[-1] != '(':
                    cls._apply_operator(output, operators.pop())
                if not operators:
                    raise ValueError("Mismatched parentheses")
                operators.pop()  # Remove '('
        
        # Apply remaining operators
        while operators:
            if operators[-1] in '()':
                raise ValueError("Mismatched parentheses")
            cls._apply_operator(output, operators.pop())
        
        if len(output) != 1:
            raise ValueError("Invalid expression")
        
        return output[0]
    
    @staticmethod
    def _is_number(token: str) -> bool:
        """Check if token is a valid number."""
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    @classmethod
    def _apply_operator(cls, output: List[float], op: str) -> None:
        """Apply operator to top two values on output stack."""
        if len(output) < 2:
            raise ValueError("Invalid expression")
        
        b = output.pop()
        a = output.pop()
        
        if op == '+':
            output.append(a + b)
        elif op == '-':
            output.append(a - b)
        elif op == '*':
            output.append(a * b)
        elif op == '/':
            output.append(a / b)
        elif op == '^':
            output.append(math.pow(a, b))
        elif op == '%':
            output.append(a % b)
    
    @staticmethod
    def _format_result(result: float) -> Union[int, str]:
        """Format result with reasonable precision."""
        # Handle infinity and NaN
        if not math.isfinite(result):
            return "Error: Result out of range"
        
        # Format with reasonable precision
        if result == int(result):
            return int(result)
        return round(result, 10)


class CalculatorTool:
    """
    Calculator tool for mathematical expressions.
    Delegates to SafeCalculator for secure evaluation.
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
        return SafeCalculator.execute(expression)


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
