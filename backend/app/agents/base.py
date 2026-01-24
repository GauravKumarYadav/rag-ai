"""
Base Agent Implementation.

Provides the foundational ReAct-style agent with a reasoning loop.
All specialized agents inherit from this base class.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar

from app.agents.state import AgentAction, AgentState, ActionType
from app.clients.lmstudio import LMStudioClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseAgent")


class BaseAgent(ABC):
    """
    ReAct-style agent with reasoning loop.
    
    Implements the Think → Act → Observe cycle:
    1. Think: Analyze current state and decide next action
    2. Act: Execute the chosen action
    3. Observe: Process results and update state
    
    Subclasses must implement:
    - think(): Analyze state and return a thought string
    - act(): Choose and return an action based on the thought
    - observe(): Process action results and update state
    """
    
    name: str = "base_agent"
    description: str = "Base agent implementation"
    
    def __init__(
        self,
        lm_client: Optional[LMStudioClient] = None,
        max_iterations: int = 3,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the base agent.
        
        Args:
            lm_client: LLM client for reasoning (optional for some agents)
            max_iterations: Maximum reasoning iterations
            verbose: If True, log detailed reasoning trace
        """
        self.lm_client = lm_client
        self.max_iterations = max_iterations
        self.verbose = verbose
    
    async def run(self, state: AgentState) -> AgentState:
        """
        Execute the agent's reasoning loop.
        
        The loop continues until:
        - A terminal action is reached
        - Maximum iterations exceeded
        - An error occurs
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state after reasoning
        """
        state.max_iterations = self.max_iterations
        start_time = time.time()
        
        logger.info(f"[{self.name}] Starting reasoning loop")
        
        while state.can_continue():
            state.increment_iteration()
            
            try:
                # Step 1: Think - Analyze current state
                thought = await self.think(state)
                state.add_thought(thought)
                
                if self.verbose:
                    logger.debug(f"[{self.name}] Thought: {thought}")
                
                # Step 2: Act - Choose and execute action
                action = await self.act(state, thought)
                state.add_action(f"{action.type.value}: {action.reasoning}")
                
                if self.verbose:
                    logger.debug(f"[{self.name}] Action: {action.type.value}")
                
                # Check for terminal action
                if action.is_terminal:
                    logger.info(f"[{self.name}] Terminal action reached: {action.reasoning}")
                    break
                
                # Step 3: Observe - Execute action and process results
                state = await self.observe(state, action)
                
                if self.verbose:
                    logger.debug(f"[{self.name}] State after observation: {state.get_context_summary()}")
                
            except Exception as e:
                logger.error(f"[{self.name}] Error in reasoning loop: {e}")
                state.add_observation(f"Error: {str(e)}")
                break
        
        elapsed = time.time() - start_time
        logger.info(f"[{self.name}] Completed in {elapsed:.2f}s, {state.iteration} iterations")
        
        return state
    
    @abstractmethod
    async def think(self, state: AgentState) -> str:
        """
        Analyze the current state and produce a thought.
        
        This is the reasoning step where the agent decides what to do next
        based on the current context, query, and any previous results.
        
        Args:
            state: Current agent state
            
        Returns:
            A thought string describing the analysis
        """
        pass
    
    @abstractmethod
    async def act(self, state: AgentState, thought: str) -> AgentAction:
        """
        Choose an action based on the current state and thought.
        
        Args:
            state: Current agent state
            thought: The thought from the think step
            
        Returns:
            An AgentAction to execute
        """
        pass
    
    @abstractmethod
    async def observe(self, state: AgentState, action: AgentAction) -> AgentState:
        """
        Execute the action and update state with results.
        
        Args:
            state: Current agent state
            action: The action to execute
            
        Returns:
            Updated agent state
        """
        pass
    
    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        """
        Helper to call the LLM for reasoning.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Generation temperature
            
        Returns:
            LLM response text
        """
        if self.lm_client is None:
            raise ValueError(f"{self.name} requires an LLM client but none was provided")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.lm_client.chat(messages, stream=False)
        return response
    
    async def _call_llm_structured(
        self,
        prompt: str,
        response_model: type,
        system_prompt: Optional[str] = None,
    ) -> Any:
        """
        Call LLM and parse response into a structured format.
        
        Uses JSON mode to get structured output from the LLM.
        
        Args:
            prompt: User prompt with JSON instructions
            response_model: Pydantic model to parse response into
            system_prompt: Optional system prompt
            
        Returns:
            Parsed response as the specified model type
        """
        import json
        import re
        
        response = await self._call_llm(prompt, system_prompt)
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return response_model(**data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse structured response: {e}")
        
        # Return default if parsing fails
        return response_model()


class SimpleAgent(BaseAgent):
    """
    A simple agent that executes a single action without a loop.
    
    Useful for straightforward tasks that don't require iterative reasoning.
    """
    
    name: str = "simple_agent"
    description: str = "Single-action agent"
    
    async def run(self, state: AgentState) -> AgentState:
        """Execute a single think → act → observe cycle."""
        try:
            thought = await self.think(state)
            state.add_thought(thought)
            
            action = await self.act(state, thought)
            state.add_action(f"{action.type.value}: {action.reasoning}")
            
            if not action.is_terminal:
                state = await self.observe(state, action)
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            state.add_observation(f"Error: {str(e)}")
        
        return state
