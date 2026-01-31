"""
Cost Tracker for LLM Usage

Calculates equivalent OpenAI API costs for local LM Studio usage.
This helps track potential savings when using local models.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from langchain_core.callbacks import AsyncCallbackHandler
from langsmith import Client as LangSmithClient

logger = logging.getLogger(__name__)


# OpenAI Pricing as of January 2025 (per 1M tokens)
# Reference: https://openai.com/pricing
OPENAI_PRICING = {
    # GPT-4o models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    
    # GPT-4 Turbo
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4-1106-preview": {"input": 10.00, "output": 30.00},
    
    # GPT-4
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-32k": {"input": 60.00, "output": 120.00},
    
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-instruct": {"input": 1.50, "output": 2.00},
    
    # Default fallback (use GPT-4o-mini pricing as baseline)
    "default": {"input": 0.15, "output": 0.60},
}


@dataclass
class TokenUsage:
    """Tracks token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = "default"
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CostSummary:
    """Summary of costs for a session or request."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    equivalent_cost_usd: float = 0.0
    savings_usd: float = 0.0  # Since we're using local model, this equals equivalent_cost
    model_used: str = "local"
    comparison_model: str = "gpt-4o-mini"
    requests_count: int = 0


def get_pricing(model: str) -> Dict[str, float]:
    """Get pricing for a model, with fallback to default."""
    # Try exact match
    if model in OPENAI_PRICING:
        return OPENAI_PRICING[model]
    
    # Try prefix match (for versioned models)
    for key in OPENAI_PRICING:
        if model.startswith(key):
            return OPENAI_PRICING[key]
    
    return OPENAI_PRICING["default"]


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o-mini"
) -> float:
    """
    Calculate the equivalent OpenAI cost for token usage.
    
    Args:
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        model: OpenAI model to compare against
        
    Returns:
        Cost in USD
    """
    pricing = get_pricing(model)
    
    # Convert from per-million to per-token
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost


class CostTrackingCallback(AsyncCallbackHandler):
    """
    LangChain callback handler that tracks token usage and calculates costs.
    Reports metrics to LangSmith.
    """
    
    def __init__(
        self,
        comparison_model: str = "gpt-4o-mini",
        langsmith_client: Optional[LangSmithClient] = None,
    ):
        self.comparison_model = comparison_model
        self.langsmith_client = langsmith_client
        self.current_run_id: Optional[str] = None
        self.usage_log: list[TokenUsage] = []
        self._lock = asyncio.Lock()
        
    @property
    def total_cost(self) -> float:
        """Calculate total equivalent cost from all logged usage."""
        return sum(
            calculate_cost(u.prompt_tokens, u.completion_tokens, self.comparison_model)
            for u in self.usage_log
        )
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return sum(u.total_tokens for u in self.usage_log)
    
    def get_summary(self) -> CostSummary:
        """Get a summary of all tracked costs."""
        total_input = sum(u.prompt_tokens for u in self.usage_log)
        total_output = sum(u.completion_tokens for u in self.usage_log)
        total = sum(u.total_tokens for u in self.usage_log)
        cost = calculate_cost(total_input, total_output, self.comparison_model)
        
        return CostSummary(
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total,
            equivalent_cost_usd=cost,
            savings_usd=cost,  # Using local model = 100% savings
            model_used="local/lmstudio",
            comparison_model=self.comparison_model,
            requests_count=len(self.usage_log),
        )
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs):
        """Called when LLM starts."""
        self.current_run_id = kwargs.get("run_id")
    
    async def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes. Extract token usage and calculate cost."""
        run_id = kwargs.get("run_id")
        
        try:
            # Extract token usage from response
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage", {})
            
            # Also check response metadata (some models put it there)
            if not token_usage and response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        if hasattr(gen, "generation_info") and gen.generation_info:
                            token_usage = gen.generation_info.get("token_usage", token_usage)
                        if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                            usage_meta = gen.message.usage_metadata
                            if usage_meta:
                                token_usage = {
                                    "prompt_tokens": getattr(usage_meta, "input_tokens", 0),
                                    "completion_tokens": getattr(usage_meta, "output_tokens", 0),
                                    "total_tokens": getattr(usage_meta, "total_tokens", 0),
                                }
            
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)
            
            # If no token count, estimate based on character count
            if total_tokens == 0 and response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        text = gen.text if hasattr(gen, "text") else str(gen)
                        # Rough estimate: ~4 chars per token
                        completion_tokens += len(text) // 4
                total_tokens = prompt_tokens + completion_tokens
            
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=llm_output.get("model_name", "local"),
            )
            
            async with self._lock:
                self.usage_log.append(usage)
            
            # Calculate equivalent cost
            cost = calculate_cost(prompt_tokens, completion_tokens, self.comparison_model)
            
            # Log the cost
            logger.info(
                f"LLM call completed: {total_tokens} tokens "
                f"(input: {prompt_tokens}, output: {completion_tokens}), "
                f"equivalent {self.comparison_model} cost: ${cost:.6f}"
            )
            
            # Update LangSmith run with cost metadata if available
            if self.langsmith_client and run_id:
                try:
                    self.langsmith_client.update_run(
                        run_id=run_id,
                        extra={
                            "metadata": {
                                "token_usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": total_tokens,
                                },
                                "cost": {
                                    "equivalent_usd": cost,
                                    "comparison_model": self.comparison_model,
                                    "actual_cost": 0.0,  # Local model is free
                                    "savings_usd": cost,
                                },
                            }
                        },
                    )
                except Exception as e:
                    logger.debug(f"Failed to update LangSmith run with cost: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to track token usage: {e}")
    
    async def on_llm_error(self, error: Exception, **kwargs):
        """Called when LLM errors."""
        logger.error(f"LLM error: {error}")
    
    def reset(self):
        """Reset the usage log."""
        self.usage_log = []


# Global cost tracker instance
_global_tracker: Optional[CostTrackingCallback] = None


def get_cost_tracker(comparison_model: str = None) -> CostTrackingCallback:
    """Get or create the global cost tracker."""
    global _global_tracker
    if _global_tracker is None:
        # Import settings here to avoid circular imports
        from app.config import settings
        
        # Use config value if not specified
        if comparison_model is None:
            comparison_model = getattr(
                settings.agent, 
                "cost_comparison_model", 
                "gpt-4o-mini"
            )
        
        try:
            langsmith_client = LangSmithClient()
        except Exception:
            langsmith_client = None
        _global_tracker = CostTrackingCallback(
            comparison_model=comparison_model,
            langsmith_client=langsmith_client,
        )
    return _global_tracker


def get_session_cost_summary() -> Dict[str, Any]:
    """Get the current session's cost summary as a dictionary."""
    tracker = get_cost_tracker()
    summary = tracker.get_summary()
    return {
        "total_input_tokens": summary.total_input_tokens,
        "total_output_tokens": summary.total_output_tokens,
        "total_tokens": summary.total_tokens,
        "equivalent_cost_usd": round(summary.equivalent_cost_usd, 6),
        "savings_usd": round(summary.savings_usd, 6),
        "model_used": summary.model_used,
        "comparison_model": summary.comparison_model,
        "requests_count": summary.requests_count,
    }
