"""
Calculator Tool.

Safe mathematical expression evaluation.
"""

import ast
import logging
import math
import operator
from typing import Any, Dict

from app.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


# Safe operators for expression evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe math functions
SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
}

# Safe constants
SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "inf": float("inf"),
}


class SafeEvaluator(ast.NodeVisitor):
    """
    Safe AST-based expression evaluator.
    
    Only allows basic arithmetic operations and safe math functions.
    Prevents code execution attacks.
    """
    
    def __init__(self):
        self.functions = SAFE_FUNCTIONS.copy()
        self.constants = SAFE_CONSTANTS.copy()
    
    def visit_Expression(self, node):
        return self.visit(node.body)
    
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    
    def visit_Num(self, node):  # Python 3.7 compatibility
        return node.n
    
    def visit_Name(self, node):
        name = node.id.lower()
        if name in self.constants:
            return self.constants[name]
        raise ValueError(f"Unknown variable: {node.id}")
    
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = SAFE_OPERATORS.get(type(node.op))
        
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        
        return op(left, right)
    
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op = SAFE_OPERATORS.get(type(node.op))
        
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        
        return op(operand)
    
    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        
        func_name = node.func.id.lower()
        if func_name not in self.functions:
            raise ValueError(f"Unknown function: {func_name}")
        
        args = [self.visit(arg) for arg in node.args]
        return self.functions[func_name](*args)
    
    def visit_List(self, node):
        return [self.visit(elem) for elem in node.elts]
    
    def visit_Tuple(self, node):
        return tuple(self.visit(elem) for elem in node.elts)
    
    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_eval(expression: str) -> Any:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression string
        
    Returns:
        Evaluated result
        
    Raises:
        ValueError: If expression contains unsafe constructs
    """
    try:
        # Parse the expression
        tree = ast.parse(expression, mode='eval')
        
        # Evaluate safely
        evaluator = SafeEvaluator()
        result = evaluator.visit(tree)
        
        return result
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    except Exception as e:
        raise ValueError(f"Evaluation error: {e}")


class CalculatorTool(BaseTool):
    """
    Mathematical calculator tool.
    
    Safely evaluates mathematical expressions including:
    - Basic arithmetic: +, -, *, /, //, %, **
    - Functions: sqrt, sin, cos, tan, log, exp, abs, round, min, max
    - Constants: pi, e
    """
    
    name = "calculator"
    description = "Perform mathematical calculations. Supports basic arithmetic, trigonometry, logarithms, and common math functions."
    parameters = [
        ToolParameter(
            name="expression",
            description="Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', '100 * 0.15')",
            type="string",
            required=True
        )
    ]
    requires_client_scope = False
    
    async def execute(self, expression: str, **kwargs) -> ToolResult:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: The math expression to evaluate
            
        Returns:
            ToolResult with the calculated value
        """
        try:
            # Clean up expression
            expression = expression.strip()
            
            # Evaluate
            result = safe_eval(expression)
            
            # Format result
            if isinstance(result, float):
                # Round to reasonable precision
                if result == int(result):
                    result = int(result)
                else:
                    result = round(result, 10)
            
            return ToolResult.ok(
                result,
                expression=expression,
                type=type(result).__name__
            )
            
        except ValueError as e:
            return ToolResult.fail(str(e))
        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return ToolResult.fail(f"Calculation failed: {str(e)}")
