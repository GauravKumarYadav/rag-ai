"""
Agentic RAG Framework.

This module provides a multi-agent orchestration system for enhanced RAG capabilities:
- Query decomposition
- Multi-hop retrieval
- Self-correction loops
- Tool calling
- Adaptive retrieval strategies

Key Components:
- AgentState: Shared state across agent reasoning steps
- BaseAgent: ReAct-style agent with reasoning loop
- OrchestratorAgent: Routes queries to specialized sub-agents
- QueryDecomposer: Breaks complex queries into sub-queries
- RetrievalAgent: Adaptive multi-hop retrieval
- SynthesisAgent: Combines results into final answer
- VerificationAgent: Self-correction with re-retrieval triggers
- ToolAgent: Tool selection and execution

Usage:
    from app.agents import get_orchestrator_agent, AgentState
    
    # Create orchestrator with all sub-agents
    orchestrator = get_orchestrator_agent(lm_client)
    
    # Initialize state
    state = AgentState(
        query="Compare revenue of company A and B",
        client_id="client_123"
    )
    
    # Run orchestration
    result = await orchestrator.run(state)
    print(result.final_answer)
"""

from app.agents.state import (
    AgentState,
    AgentAction,
    AgentPlan,
    SubQuery,
    CoverageAssessment,
    VerificationResult,
    ActionType,
)
from app.agents.base import BaseAgent, SimpleAgent
from app.agents.orchestrator import OrchestratorAgent, get_orchestrator_agent
from app.agents.query_decomposer import QueryDecomposer, get_query_decomposer
from app.agents.retrieval_agent import RetrievalAgent, get_retrieval_agent
from app.agents.synthesis_agent import SynthesisAgent, get_synthesis_agent
from app.agents.verification_agent import VerificationAgentImpl, get_verification_agent
from app.agents.tool_agent import ToolAgent, get_tool_agent
from app.agents.model_router import (
    ModelRouter,
    ModelTier,
    TaskType,
    get_model_router,
    select_model_for_task,
)

__all__ = [
    # State management
    "AgentState",
    "AgentAction",
    "AgentPlan",
    "SubQuery",
    "CoverageAssessment",
    "VerificationResult",
    "ActionType",
    
    # Base classes
    "BaseAgent",
    "SimpleAgent",
    
    # Agents
    "OrchestratorAgent",
    "QueryDecomposer",
    "RetrievalAgent",
    "SynthesisAgent",
    "VerificationAgentImpl",
    "ToolAgent",
    
    # Factory functions
    "get_orchestrator_agent",
    "get_query_decomposer",
    "get_retrieval_agent",
    "get_synthesis_agent",
    "get_verification_agent",
    "get_tool_agent",
    
    # Model routing
    "ModelRouter",
    "ModelTier",
    "TaskType",
    "get_model_router",
    "select_model_for_task",
]
