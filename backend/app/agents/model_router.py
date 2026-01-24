"""
Model Router.

Routes tasks to appropriate models based on complexity.
Supports flexible switching between small and large models.
"""

import logging
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks that need different model capabilities."""
    PLANNING = "planning"
    SYNTHESIS = "synthesis"
    COMPLEX_REASONING = "complex_reasoning"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    SIMPLE_QA = "simple_qa"
    DECOMPOSITION = "decomposition"
    VERIFICATION = "verification"


class ModelTier(str, Enum):
    """Model capability tiers."""
    FAST = "fast"          # Small, fast models (1-7B params)
    CAPABLE = "capable"    # Medium, capable models (7-30B params)
    POWERFUL = "powerful"  # Large, powerful models (30B+ or API)


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str
    provider: str  # lmstudio, ollama, openai, custom
    tier: ModelTier
    context_window: int = 8000
    supports_json: bool = True
    supports_vision: bool = False


# Default task-to-tier mapping
DEFAULT_TASK_ROUTING = {
    TaskType.PLANNING: ModelTier.CAPABLE,
    TaskType.SYNTHESIS: ModelTier.CAPABLE,
    TaskType.COMPLEX_REASONING: ModelTier.POWERFUL,
    TaskType.EXTRACTION: ModelTier.FAST,
    TaskType.CLASSIFICATION: ModelTier.FAST,
    TaskType.SIMPLE_QA: ModelTier.FAST,
    TaskType.DECOMPOSITION: ModelTier.CAPABLE,
    TaskType.VERIFICATION: ModelTier.FAST,
}


class ModelRouter:
    """
    Routes tasks to appropriate models based on complexity.
    
    Features:
    - Task-based routing (planning vs extraction vs synthesis)
    - Tier-based fallback (if capable not available, use fast)
    - Dynamic model registration
    - Config-driven model selection
    """
    
    def __init__(
        self,
        fast_model: Optional[str] = None,
        capable_model: Optional[str] = None,
        powerful_model: Optional[str] = None,
        use_routing: bool = True,
    ) -> None:
        """
        Initialize model router.
        
        Args:
            fast_model: Name of fast model (e.g., "llama3.2:1b")
            capable_model: Name of capable model (e.g., "qwen3-vl-30b")
            powerful_model: Name of powerful model (e.g., "gpt-4")
            use_routing: If False, always use capable model
        """
        self.use_routing = use_routing
        
        # Set default models from config
        self._models: Dict[ModelTier, str] = {}
        
        if fast_model:
            self._models[ModelTier.FAST] = fast_model
        if capable_model:
            self._models[ModelTier.CAPABLE] = capable_model
        if powerful_model:
            self._models[ModelTier.POWERFUL] = powerful_model
        
        # Task routing configuration
        self._task_routing = DEFAULT_TASK_ROUTING.copy()
        
        # Model configurations
        self._model_configs: Dict[str, ModelConfig] = {}
    
    def register_model(self, config: ModelConfig) -> None:
        """
        Register a model configuration.
        
        Args:
            config: Model configuration
        """
        self._model_configs[config.name] = config
        
        # Update tier mapping if not already set
        if config.tier not in self._models:
            self._models[config.tier] = config.name
        
        logger.debug(f"Registered model: {config.name} (tier: {config.tier.value})")
    
    def set_task_routing(self, task: TaskType, tier: ModelTier) -> None:
        """
        Configure routing for a specific task type.
        
        Args:
            task: Task type to configure
            tier: Model tier to route to
        """
        self._task_routing[task] = tier
    
    def select_model(self, task: TaskType) -> str:
        """
        Select the appropriate model for a task.
        
        Args:
            task: Type of task to perform
            
        Returns:
            Model name to use
        """
        if not self.use_routing:
            # Always use capable model when routing disabled
            return self._get_model_for_tier(ModelTier.CAPABLE)
        
        # Get preferred tier for task
        preferred_tier = self._task_routing.get(task, ModelTier.FAST)
        
        # Try to get model for preferred tier, fall back if not available
        return self._get_model_for_tier(preferred_tier)
    
    def _get_model_for_tier(self, tier: ModelTier) -> str:
        """
        Get model name for a tier with fallback.
        
        Fallback order: preferred -> capable -> fast -> any available
        """
        # Try preferred tier
        if tier in self._models:
            return self._models[tier]
        
        # Fall back in order
        fallback_order = [ModelTier.CAPABLE, ModelTier.FAST, ModelTier.POWERFUL]
        
        for fallback_tier in fallback_order:
            if fallback_tier in self._models:
                logger.debug(f"Falling back from {tier.value} to {fallback_tier.value}")
                return self._models[fallback_tier]
        
        # No models available
        logger.warning("No models configured in router")
        return "default"
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self._model_configs.get(model_name)
    
    def get_tier_for_task(self, task: TaskType) -> ModelTier:
        """Get the model tier assigned to a task."""
        return self._task_routing.get(task, ModelTier.FAST)
    
    def list_models(self) -> Dict[str, str]:
        """List all configured models by tier."""
        return {tier.value: model for tier, model in self._models.items()}
    
    def estimate_cost(self, task: TaskType, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for a task (if using API models).
        
        Returns 0 for local models.
        """
        model = self.select_model(task)
        config = self.get_model_config(model)
        
        if config is None or config.provider not in ["openai", "anthropic"]:
            return 0.0
        
        # Rough cost estimates per 1K tokens
        COST_PER_1K = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4o-mini": {"input": 0.0015, "output": 0.006},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }
        
        rates = COST_PER_1K.get(model, {"input": 0.001, "output": 0.002})
        
        cost = (input_tokens / 1000 * rates["input"]) + (output_tokens / 1000 * rates["output"])
        return cost


# Global router instance
_router: Optional[ModelRouter] = None


def get_model_router() -> ModelRouter:
    """
    Get or create the global model router.
    
    Configures from settings if available.
    """
    global _router
    
    if _router is None:
        # Try to load from config
        fast_model = None
        capable_model = None
        powerful_model = None
        use_routing = True
        
        try:
            from app.config import settings
            
            if hasattr(settings, 'agent'):
                fast_model = getattr(settings.agent, 'fast_model', None)
                capable_model = getattr(settings.agent, 'capable_model', None)
                use_routing = getattr(settings.agent, 'use_model_routing', True)
            
            # Fall back to LLM settings
            if not capable_model:
                capable_model = settings.llm.get_active_provider().model
                
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not load config for model router: {e}")
            capable_model = "default"
        
        _router = ModelRouter(
            fast_model=fast_model or capable_model,
            capable_model=capable_model,
            powerful_model=powerful_model,
            use_routing=use_routing,
        )
        
        logger.info(f"Model router initialized: {_router.list_models()}")
    
    return _router


def select_model_for_task(task: TaskType) -> str:
    """
    Convenience function to select model for a task.
    """
    return get_model_router().select_model(task)
