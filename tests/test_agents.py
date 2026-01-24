"""
Tests for the Agentic RAG Framework.

Tests cover:
- Agent state management
- Query decomposition
- Tool execution
- Model routing
- Integration tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.state import (
    AgentState,
    AgentAction,
    AgentPlan,
    SubQuery,
    ActionType,
    CoverageAssessment,
)
from app.agents.base import BaseAgent
from app.agents.query_decomposer import QueryDecomposer
from app.agents.model_router import ModelRouter, ModelTier, TaskType
from app.tools.base import BaseTool, ToolResult, ToolParameter
from app.tools.registry import ToolRegistry
from app.tools.calculator import CalculatorTool, safe_eval
from app.tools.datetime_tool import DateTimeTool
from app.services.natural_responses import llm_unavailable_response
from app.services.document_listing import list_documents
from app.rag.retriever import resolve_retrieval_scopes, _merge_hits_by_scope
from app.services.response_generator import generate_response


class TestAgentState:
    """Tests for AgentState."""
    
    def test_state_initialization(self):
        """Test basic state initialization."""
        state = AgentState(query="What is the revenue?")
        
        assert state.query == "What is the revenue?"
        assert state.iteration == 0
        assert state.client_id is None
        assert len(state.retrieved_context) == 0
        assert len(state.scratchpad) == 0
    
    def test_state_with_client_isolation(self):
        """Test state preserves client isolation."""
        state = AgentState(
            query="Test query",
            client_id="client_123",
            allowed_clients=["client_123", "client_456"]
        )
        
        assert state.client_id == "client_123"
        assert "client_123" in state.allowed_clients
    
    def test_add_thought(self):
        """Test adding thoughts to scratchpad."""
        state = AgentState(query="Test")
        state.add_thought("Need to retrieve documents")
        
        assert len(state.scratchpad) == 1
        assert "Thought:" in state.scratchpad[0]
    
    def test_can_continue(self):
        """Test iteration limits."""
        state = AgentState(query="Test", max_iterations=3, max_corrections=2)
        
        assert state.can_continue() is True
        
        state.iteration = 3
        assert state.can_continue() is False
        
        state.iteration = 0
        state.correction_attempts = 2
        assert state.can_continue() is False
    
    def test_merge_retrieved_deduplicates(self):
        """Test that merge_retrieved removes duplicates."""
        from app.models.schemas import RetrievalHit
        
        state = AgentState(query="Test")
        
        hit1 = RetrievalHit(id="doc1", content="Content 1", score=0.9)
        hit2 = RetrievalHit(id="doc2", content="Content 2", score=0.8)
        hit3 = RetrievalHit(id="doc1", content="Content 1 duplicate", score=0.7)
        
        state.merge_retrieved([hit1, hit2])
        assert len(state.retrieved_context) == 2
        
        state.merge_retrieved([hit3])  # Duplicate ID
        assert len(state.retrieved_context) == 2  # No duplicate added


class TestAgentAction:
    """Tests for AgentAction."""
    
    def test_stop_action(self):
        """Test creating a stop action."""
        action = AgentAction.stop("Task complete")
        
        assert action.type == ActionType.STOP
        assert action.is_terminal is True
        assert "complete" in action.reasoning.lower()
    
    def test_retrieve_action(self):
        """Test creating a retrieve action."""
        action = AgentAction.retrieve("revenue data", strategy="hybrid")
        
        assert action.type == ActionType.RETRIEVE
        assert action.params["query"] == "revenue data"
        assert action.params["strategy"] == "hybrid"
    
    def test_use_tool_action(self):
        """Test creating a tool use action."""
        action = AgentAction.use_tool("calculator", {"expression": "2 + 2"})
        
        assert action.type == ActionType.USE_TOOL
        assert action.params["tool"] == "calculator"


class TestSubQuery:
    """Tests for SubQuery."""
    
    def test_subquery_creation(self):
        """Test creating a sub-query."""
        sq = SubQuery(
            query="What is revenue for Q1?",
            purpose="Get Q1 data for comparison",
            retrieval_hints=["revenue", "Q1", "2024"]
        )
        
        assert sq.query == "What is revenue for Q1?"
        assert sq.executed is False
        assert len(sq.retrieval_hints) == 3
    
    def test_subquery_dependencies(self):
        """Test sub-query dependencies."""
        sq1 = SubQuery(query="What is X?")
        sq2 = SubQuery(query="Compare X and Y", depends_on=[0])
        
        assert len(sq2.depends_on) == 1
        assert 0 in sq2.depends_on


class TestQueryDecomposer:
    """Tests for QueryDecomposer."""
    
    def test_complexity_assessment_simple(self):
        """Test complexity detection for simple queries."""
        mock_client = MagicMock()
        decomposer = QueryDecomposer(lm_client=mock_client)
        
        # Simple query
        score = decomposer._assess_complexity("What is the capital of France?")
        assert score < 0.3
    
    def test_complexity_assessment_complex(self):
        """Test complexity detection for complex queries."""
        mock_client = MagicMock()
        decomposer = QueryDecomposer(lm_client=mock_client)
        
        # Comparison query (with "compare" keyword)
        score = decomposer._assess_complexity("Compare the revenue of company A versus company B")
        assert score >= 0.25, f"Expected >= 0.25, got {score}"
        
        # Temporal query with explicit change indicator
        score = decomposer._assess_complexity("How did sales change between Q1 2024 and Q2 2024?")
        assert score >= 0.25, f"Expected >= 0.25, got {score}"
        
        # Multi-part question
        score = decomposer._assess_complexity("What are all the products and also list their prices?")
        assert score >= 0.3, f"Expected >= 0.3, got {score}"


class TestCalculatorTool:
    """Tests for CalculatorTool."""
    
    def test_safe_eval_basic(self):
        """Test basic arithmetic."""
        assert safe_eval("2 + 2") == 4
        assert safe_eval("10 - 3") == 7
        assert safe_eval("4 * 5") == 20
        assert safe_eval("15 / 3") == 5
    
    def test_safe_eval_complex(self):
        """Test complex expressions."""
        assert safe_eval("2 ** 3") == 8
        assert safe_eval("(2 + 3) * 4") == 20
        assert abs(safe_eval("sqrt(16)") - 4.0) < 0.001
    
    def test_safe_eval_functions(self):
        """Test math functions."""
        import math
        
        assert abs(safe_eval("sin(0)") - 0) < 0.001
        assert abs(safe_eval("cos(0)") - 1) < 0.001
        assert abs(safe_eval("log(e)") - 1) < 0.001
    
    def test_safe_eval_constants(self):
        """Test constants."""
        import math
        
        assert abs(safe_eval("pi") - math.pi) < 0.001
        assert abs(safe_eval("e") - math.e) < 0.001
    
    def test_safe_eval_rejects_dangerous(self):
        """Test that dangerous operations are rejected."""
        with pytest.raises(ValueError):
            safe_eval("__import__('os').system('ls')")
        
        with pytest.raises(ValueError):
            safe_eval("open('/etc/passwd')")
    
    @pytest.mark.asyncio
    async def test_calculator_tool_execute(self):
        """Test calculator tool execution."""
        tool = CalculatorTool()
        
        result = await tool.execute(expression="100 * 0.15")
        
        assert result.success is True
        assert result.result == 15


class TestDateTimeTool:
    """Tests for DateTimeTool."""
    
    @pytest.mark.asyncio
    async def test_datetime_now(self):
        """Test getting current datetime."""
        tool = DateTimeTool()
        
        result = await tool.execute(operation="now")
        
        assert result.success is True
        assert result.result is not None
    
    @pytest.mark.asyncio
    async def test_datetime_parse(self):
        """Test parsing dates."""
        tool = DateTimeTool()
        
        result = await tool.execute(operation="parse", date="2024-01-15")
        
        assert result.success is True
        assert "2024-01-15" in str(result.result)
    
    @pytest.mark.asyncio
    async def test_datetime_add(self):
        """Test adding days to date."""
        tool = DateTimeTool()
        
        result = await tool.execute(
            operation="add",
            date="2024-01-15",
            days=10,
            format="%Y-%m-%d"
        )
        
        assert result.success is True
        assert "2024-01-25" in str(result.result)
    
    @pytest.mark.asyncio
    async def test_datetime_diff(self):
        """Test date difference."""
        tool = DateTimeTool()
        
        result = await tool.execute(
            operation="diff",
            date="2024-01-01",
            date2="2024-01-31"
        )
        
        assert result.success is True
        assert result.metadata["total_days"] == 30


class TestToolRegistry:
    """Tests for ToolRegistry."""
    
    def test_register_tool(self):
        """Test tool registration."""
        registry = ToolRegistry()
        tool = CalculatorTool()
        
        registry.register(tool)
        
        assert "calculator" in registry.list_tools()
        assert registry.get("calculator") is tool
    
    def test_get_tool_descriptions(self):
        """Test getting tool descriptions for prompts."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(DateTimeTool())
        
        descriptions = registry.get_tool_descriptions()
        
        assert "calculator" in descriptions.lower()
        assert "datetime" in descriptions.lower()
    
    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing tool through registry."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        
        result = await registry.execute("calculator", expression="5 + 5")
        
        assert result.success is True
        assert result.result == 10
    
    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing unknown tool returns error."""
        registry = ToolRegistry()
        
        result = await registry.execute("unknown_tool")
        
        assert result.success is False
        assert "not found" in result.error.lower()


class TestToolAgent:
    """Tests for ToolAgent behavior."""
    
    @pytest.mark.asyncio
    async def test_tool_agent_skips_document_list_query(self):
        """Tool agent should skip document list queries."""
        mock_client = MagicMock()
        from app.agents.tool_agent import ToolAgent
        
        agent = ToolAgent(lm_client=mock_client)
        state = AgentState(query="list documents")
        
        result = await agent.run(state)
        assert result.tool_results == {}


class TestRetrievalScopes:
    """Tests for scope resolution and merging."""
    
    def test_resolve_retrieval_scopes(self):
        assert resolve_retrieval_scopes(None) == ["global"]
        assert resolve_retrieval_scopes("global") == ["global"]
        assert resolve_retrieval_scopes("client1", ["client1"]) == ["client1", "global"]
        assert resolve_retrieval_scopes("client2", ["client1"]) == ["global"]
    
    def test_merge_hits_prefers_lower_score(self):
        from app.models.schemas import RetrievalHit
        
        global_hit = RetrievalHit(id="doc1", content="global", score=0.6, metadata={})
        client_hit = RetrievalHit(id="doc1", content="client", score=0.4, metadata={})
        
        merged = _merge_hits_by_scope({
            "global": [global_hit],
            "client1": [client_hit],
        })
        
        assert len(merged) == 1
        assert merged[0].content == "client"
        assert merged[0].metadata.get("scope") == "client1"


class TestDocumentListingScopes:
    """Tests for document listing scope aggregation."""
    
    def test_list_documents_includes_global(self, monkeypatch):
        class FakeDocs:
            def __init__(self, ids, metadatas):
                self._ids = ids
                self._metadatas = metadatas

            def get(self, include=None):
                return {"ids": self._ids, "metadatas": self._metadatas}

        class FakeStore:
            def __init__(self, ids, metadatas):
                self.docs = FakeDocs(ids, metadatas)

        def fake_global_store():
            return FakeStore(
                ["g1"],
                [{"source": "/global/global.pdf", "client_id": "global"}],
            )

        def fake_client_store(client_id):
            return FakeStore(
                ["c1"],
                [{"source": f"/clients/{client_id}/client.pdf", "client_id": client_id}],
            )

        monkeypatch.setattr(
            "app.services.document_listing.get_vector_store",
            fake_global_store,
        )
        monkeypatch.setattr(
            "app.services.document_listing.get_client_vector_store",
            fake_client_store,
        )

        documents = list_documents(client_id="client1", include_global=True)
        filenames = {doc["filename"] for doc in documents}

        assert "global.pdf" in filenames
        assert "client.pdf" in filenames

    def test_list_documents_global_only(self, monkeypatch):
        class FakeDocs:
            def __init__(self, ids, metadatas):
                self._ids = ids
                self._metadatas = metadatas

            def get(self, include=None):
                return {"ids": self._ids, "metadatas": self._metadatas}

        class FakeStore:
            def __init__(self, ids, metadatas):
                self.docs = FakeDocs(ids, metadatas)

        def fake_global_store():
            return FakeStore(
                ["g1"],
                [{"source": "/global/global.pdf", "client_id": "global"}],
            )

        monkeypatch.setattr(
            "app.services.document_listing.get_vector_store",
            fake_global_store,
        )

        documents = list_documents(client_id="global", include_global=False)
        filenames = {doc["filename"] for doc in documents}

        assert filenames == {"global.pdf"}


class TestResponseGenerator:
    """Tests for LLM response generator fallbacks."""
    
    @pytest.mark.asyncio
    async def test_generate_response_without_llm(self):
        """Test generator returns LLM-unavailable response."""
        response = await generate_response(
            lm_client=None,
            response_type="chitchat",
            user_message="hi",
        )
        assert response == llm_unavailable_response()


class TestModelRouter:
    """Tests for ModelRouter."""
    
    def test_model_selection_by_task(self):
        """Test model selection based on task type."""
        router = ModelRouter(
            fast_model="llama3.2:1b",
            capable_model="qwen3-vl-30b",
            use_routing=True
        )
        
        # Fast tasks
        model = router.select_model(TaskType.CLASSIFICATION)
        assert model == "llama3.2:1b"
        
        model = router.select_model(TaskType.EXTRACTION)
        assert model == "llama3.2:1b"
        
        # Capable tasks
        model = router.select_model(TaskType.PLANNING)
        assert model == "qwen3-vl-30b"
        
        model = router.select_model(TaskType.SYNTHESIS)
        assert model == "qwen3-vl-30b"
    
    def test_routing_disabled(self):
        """Test that routing disabled always uses capable model."""
        router = ModelRouter(
            fast_model="llama3.2:1b",
            capable_model="qwen3-vl-30b",
            use_routing=False
        )
        
        # All tasks should use capable model
        assert router.select_model(TaskType.CLASSIFICATION) == "qwen3-vl-30b"
        assert router.select_model(TaskType.PLANNING) == "qwen3-vl-30b"
    
    def test_fallback_when_tier_missing(self):
        """Test fallback when preferred tier is missing."""
        router = ModelRouter(
            capable_model="qwen3-vl-30b",
            use_routing=True
        )
        
        # Fast tier not configured, should fall back to capable
        model = router.select_model(TaskType.CLASSIFICATION)
        assert model == "qwen3-vl-30b"
    
    def test_list_models(self):
        """Test listing configured models."""
        router = ModelRouter(
            fast_model="llama3.2:1b",
            capable_model="qwen3-vl-30b"
        )
        
        models = router.list_models()
        
        assert "fast" in models
        assert "capable" in models


class TestAgentPlan:
    """Tests for AgentPlan."""
    
    def test_plan_creation(self):
        """Test creating an execution plan."""
        plan = AgentPlan(
            sub_queries=[
                SubQuery(query="What is X?"),
                SubQuery(query="What is Y?")
            ],
            agent_sequence=["query_decomposer", "retrieval_agent", "synthesis_agent"],
            requires_tools=False,
            estimated_complexity="moderate",
            reasoning="Query requires comparison"
        )
        
        assert len(plan.sub_queries) == 2
        assert "retrieval_agent" in plan.agent_sequence
        assert plan.estimated_complexity == "moderate"


class TestCoverageAssessment:
    """Tests for CoverageAssessment."""
    
    def test_coverage_assessment(self):
        """Test coverage assessment."""
        assessment = CoverageAssessment(
            is_sufficient=True,
            coverage_ratio=0.85,
            missing_entities=[],
            needs_kg_expansion=False,
            confidence=0.9
        )
        
        assert assessment.is_sufficient is True
        assert assessment.coverage_ratio == 0.85
    
    def test_coverage_with_missing_entities(self):
        """Test coverage with missing entities."""
        assessment = CoverageAssessment(
            is_sufficient=False,
            coverage_ratio=0.4,
            missing_entities=["entity1", "entity2"],
            needs_kg_expansion=True,
            confidence=0.4
        )
        
        assert assessment.is_sufficient is False
        assert len(assessment.missing_entities) == 2
        assert assessment.needs_kg_expansion is True


class TestIntentClassification:
    """Tests for intent classification in the agentic pipeline."""
    
    def test_state_with_intent_fields(self):
        """Test state initialization with intent fields."""
        state = AgentState(
            query="What is the revenue?",
            intent="question",
            intent_confidence=0.85,
            needs_retrieval=True,
            resolved_references=["doc1"]
        )
        
        assert state.intent == "question"
        assert state.intent_confidence == 0.85
        assert state.needs_retrieval is True
        assert "doc1" in state.resolved_references
    
    def test_state_default_intent_fields(self):
        """Test default values for intent fields."""
        state = AgentState(query="Test query")
        
        assert state.intent is None
        assert state.intent_confidence == 0.0
        assert state.needs_retrieval is True  # Default to True for safety
        assert state.resolved_references == []
        assert state.search_query == ""

    @pytest.mark.asyncio
    async def test_query_processor_document_list_intent(self):
        """Test document listing intent detection."""
        from app.services.query_processor import QueryProcessor, Intent
        from app.memory.conversation_state import ConversationState
        
        processor = QueryProcessor()
        result = await processor.process("what documents are available", ConversationState())
        
        assert result.intent == Intent.DOCUMENT_LIST
        assert result.needs_retrieval is False

    @pytest.mark.asyncio
    async def test_query_processor_clarification_intent(self):
        """Test clarification intent detection."""
        from app.services.query_processor import QueryProcessor, Intent
        from app.memory.conversation_state import ConversationState
        
        processor = QueryProcessor()
        result = await processor.process("what do you mean?", ConversationState())
        
        assert result.intent == Intent.CLARIFICATION
        assert result.needs_retrieval is False
    
    def test_is_tool_only_query_calculation(self):
        """Test tool-only detection for calculations."""
        mock_client = MagicMock()
        from app.agents.orchestrator import OrchestratorAgent
        
        orchestrator = OrchestratorAgent(lm_client=mock_client)
        
        # Calculation queries
        state = AgentState(query="What is 25 * 4?")
        assert orchestrator._is_tool_only_query(state) is True
        
        state = AgentState(query="Calculate 100 + 200")
        assert orchestrator._is_tool_only_query(state) is True
        
        state = AgentState(query="15 / 3")
        assert orchestrator._is_tool_only_query(state) is True
    
    def test_is_tool_only_query_datetime(self):
        """Test tool-only detection for date/time queries."""
        mock_client = MagicMock()
        from app.agents.orchestrator import OrchestratorAgent
        
        orchestrator = OrchestratorAgent(lm_client=mock_client)
        
        # DateTime queries
        state = AgentState(query="What is today's date?")
        assert orchestrator._is_tool_only_query(state) is True
        
        state = AgentState(query="What is the current time?")
        assert orchestrator._is_tool_only_query(state) is True
        
        state = AgentState(query="How many days between January and March?")
        assert orchestrator._is_tool_only_query(state) is True
    
    def test_is_not_tool_only_query(self):
        """Test that regular queries are not marked as tool-only."""
        mock_client = MagicMock()
        from app.agents.orchestrator import OrchestratorAgent
        
        orchestrator = OrchestratorAgent(lm_client=mock_client)
        
        # Regular queries should not be tool-only
        state = AgentState(query="What is the company's revenue?")
        assert orchestrator._is_tool_only_query(state) is False
        
        state = AgentState(query="Tell me about the project")
        assert orchestrator._is_tool_only_query(state) is False
    
    def test_has_question_component(self):
        """Test mixed intent detection."""
        mock_client = MagicMock()
        from app.agents.orchestrator import OrchestratorAgent
        
        orchestrator = OrchestratorAgent(lm_client=mock_client)
        
        # Mixed greeting + question
        assert orchestrator._has_question_component("Hi, what is the revenue?") is True
        assert orchestrator._has_question_component("Hello, show me the documents") is True
        assert orchestrator._has_question_component("Hey, find the Q1 report?") is True
        
        # Pure greetings should not have question component
        assert orchestrator._has_question_component("hi") is False
        assert orchestrator._has_question_component("hello!") is False
        assert orchestrator._has_question_component("hey there") is False
    
    def test_is_rejection_response(self):
        """Test rejection response detection."""
        mock_client = MagicMock()
        from app.agents.orchestrator import OrchestratorAgent
        
        orchestrator = OrchestratorAgent(lm_client=mock_client)
        
        # Rejection patterns
        assert orchestrator._is_rejection_response("no") is True
        assert orchestrator._is_rejection_response("nope") is True
        assert orchestrator._is_rejection_response("wrong") is True
        assert orchestrator._is_rejection_response("that's not right") is True
        assert orchestrator._is_rejection_response("try again") is True
        
        # Non-rejection patterns
        assert orchestrator._is_rejection_response("yes") is False
        assert orchestrator._is_rejection_response("ok") is False
        assert orchestrator._is_rejection_response("What is the revenue?") is False
    
    @pytest.mark.asyncio
    async def test_generate_conversational_response_greeting(self):
        """Test conversational response for greetings."""
        mock_client = MagicMock()
        from app.agents.orchestrator import OrchestratorAgent
        
        orchestrator = OrchestratorAgent(lm_client=mock_client)
        
        state = AgentState(query="hello")
        response = await orchestrator._generate_conversational_response(state)
        
        # Should return a greeting response
        assert any(word in response.lower() for word in ["hello", "hi", "help", "assist"])
    
    @pytest.mark.asyncio
    async def test_generate_conversational_response_thanks(self):
        """Test conversational response for thanks."""
        mock_client = MagicMock()
        from app.agents.orchestrator import OrchestratorAgent
        
        orchestrator = OrchestratorAgent(lm_client=mock_client)
        
        state = AgentState(query="thank you")
        response = await orchestrator._generate_conversational_response(state)
        
        # Should return an acknowledgment
        assert any(word in response.lower() for word in ["welcome", "happy", "help"])
    
    @pytest.mark.asyncio
    async def test_generate_conversational_response_help(self):
        """Test conversational response for help requests."""
        mock_client = MagicMock()
        from app.agents.orchestrator import OrchestratorAgent
        
        orchestrator = OrchestratorAgent(lm_client=mock_client)
        
        state = AgentState(query="what can you do?")
        response = await orchestrator._generate_conversational_response(state)
        
        # Should describe capabilities
        assert any(word in response.lower() for word in ["document", "search", "help", "calculation"])
    
    def test_format_tool_results_for_answer(self):
        """Test formatting tool results for response."""
        mock_client = MagicMock()
        from app.agents.orchestrator import OrchestratorAgent
        
        orchestrator = OrchestratorAgent(lm_client=mock_client)
        
        # Test calculator result
        state = AgentState(query="5 + 5", tool_results={"calculator": 10})
        response = orchestrator._format_tool_results_for_answer(state)
        assert "10" in response
        assert "result" in response.lower()
        
        # Test datetime result
        state = AgentState(query="today", tool_results={"datetime": "2024-01-15"})
        response = orchestrator._format_tool_results_for_answer(state)
        assert "2024-01-15" in response
        
        # Test empty tool results
        state = AgentState(query="test", tool_results={})
        response = orchestrator._format_tool_results_for_answer(state)
        assert "couldn't" in response.lower()


class TestSynthesisAgentNoContext:
    """Tests for SynthesisAgent no-context handling."""
    
    def test_generate_no_context_response_chitchat(self):
        """Test no-context response for chitchat intent."""
        mock_client = MagicMock()
        from app.agents.synthesis_agent import SynthesisAgent
        
        agent = SynthesisAgent(lm_client=mock_client)
        
        state = AgentState(query="hello", intent="chitchat")
        response = agent._generate_no_context_response("hello", state)
        
        assert "hello" in response.lower() or "help" in response.lower()
    
    def test_generate_no_context_response_with_tools(self):
        """Test no-context response when tools were used."""
        mock_client = MagicMock()
        from app.agents.synthesis_agent import SynthesisAgent
        
        agent = SynthesisAgent(lm_client=mock_client)
        
        state = AgentState(
            query="5 + 5",
            intent="question",
            tool_results={"calculator": 10}
        )
        response = agent._generate_no_context_response("5 + 5", state)
        
        assert "10" in response or "calculator" in response.lower()
    
    def test_generate_no_context_response_no_documents(self):
        """Test no-context response when no documents found."""
        mock_client = MagicMock()
        from app.agents.synthesis_agent import SynthesisAgent
        
        agent = SynthesisAgent(lm_client=mock_client)
        
        state = AgentState(query="What is the revenue?", intent="question")
        response = agent._generate_no_context_response("What is the revenue?", state)
        
        # Should provide helpful guidance
        assert "couldn't find" in response.lower() or "no" in response.lower()
        assert any(word in response.lower() for word in ["document", "rephrase", "upload", "question"])


class TestNaturalResponses:
    """Tests for natural response templates."""
    
    def test_format_document_list_response(self):
        """Test formatting document list responses."""
        from app.services.natural_responses import format_document_list_response
        
        documents = [
            {"filename": "alpha.pdf"},
            {"filename": "beta.docx"},
        ]
        
        response = format_document_list_response(documents)
        
        assert "alpha.pdf" in response
        assert "beta.docx" in response
        assert "documents" in response.lower()
    
    def test_format_document_list_response_empty(self):
        """Test formatting empty document list response."""
        from app.services.natural_responses import format_document_list_response
        
        response = format_document_list_response([])
        
        assert "don't see any documents" in response.lower()


# Integration test placeholder
class TestAgenticPipelineIntegration:
    """Integration tests for the full agentic pipeline."""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires LLM client - run manually")
    async def test_full_orchestration(self):
        """Test full orchestration flow."""
        from app.agents import get_orchestrator_agent, AgentState
        
        # This test requires a real LLM client
        # Skip in CI, run manually for integration testing
        pass
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires LLM client - run manually")
    async def test_query_decomposition_flow(self):
        """Test query decomposition in the pipeline."""
        pass
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires LLM client - run manually")
    async def test_chitchat_skips_retrieval(self):
        """Test that chitchat queries skip retrieval."""
        from app.agents.orchestrator import get_orchestrator_agent
        
        # Create orchestrator
        orchestrator = get_orchestrator_agent(include_all_agents=True)
        
        # Test greeting
        state = AgentState(query="hi")
        result = await orchestrator.run(state)
        
        # Should have a response without retrieval
        assert result.final_answer is not None
        assert len(result.retrieved_context) == 0
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires LLM client - run manually")
    async def test_mixed_intent_triggers_retrieval(self):
        """Test that mixed greeting+question triggers retrieval."""
        from app.agents.orchestrator import get_orchestrator_agent
        
        orchestrator = get_orchestrator_agent(include_all_agents=True)
        
        state = AgentState(query="Hi, what is the company revenue?")
        result = await orchestrator.run(state)
        
        # Should have attempted retrieval
        assert result.intent == "question"  # Upgraded from chitchat


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
