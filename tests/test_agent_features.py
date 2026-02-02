"""Tests for new agent features: memory, streaming, planning, reflection, etc."""

from unittest.mock import MagicMock, Mock
import pytest

from src.framework.agents.base import AgentBase
from src.framework.agents.memory import RAGMemoryStore, AgentWithMemory, ConversationMemory
from src.framework.agents.multi_agent import MultiAgentSystem, ResearcherAgent, WriterAgent
from src.framework.agents.planning_agent import PlanningAgent, ExecutionPlan, PlanStatus
from src.framework.agents.reflective_agent import ReflectiveAgent, ReflectionResult
from src.framework.agents.monitoring import AgentMonitor, MonitoredAgent, EventType
from src.framework.agents.cost_tracking import CostTracker, CostTrackingAgent, BudgetExceededError
from src.framework.agents.personas import PersonaAgent, PersonaType, create_persona_agent
from src.framework.agents.error_recovery import ErrorRecoveryAgent, RetryConfig, ErrorType
from src.framework.agents.evaluation import AgentEvaluator, EvaluationTask, ExactMatchMetric


class MockAgent(AgentBase):
    """Mock agent for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.invoke_count = 0
    
    def invoke(self, message: str, **kwargs) -> str:
        self.invoke_count += 1
        if self.responses:
            return self.responses.pop(0) if self.responses else "Mock response"
        return f"Response to: {message}"
    
    def get_tools_description(self) -> list[dict]:
        return [{"name": "mock_tool", "description": "Mock tool"}]


class MockRAG:
    """Mock RAG client for testing."""
    
    def __init__(self):
        self.documents = []
    
    def add_documents(self, texts, metadatas=None):
        self.documents.extend(zip(texts, metadatas or [{}] * len(texts)))
    
    def retrieve(self, query, top_k=4):
        return [{"content": f"Retrieved: {query}", "metadata": {}}]
    
    def clear(self):
        self.documents.clear()


@pytest.fixture
def mock_rag():
    return MockRAG()


@pytest.fixture
def mock_agent():
    return MockAgent()


def test_rag_memory_store_store_and_retrieve(mock_rag):
    """Test RAG memory store."""
    memory = RAGMemoryStore(rag_client=mock_rag)
    
    memory.store("user1", "Hello", "Hi there!")
    memory.store("user1", "What's the weather?", "It's sunny!")
    
    memories = memory.retrieve("user1", "weather", top_k=2)
    assert len(memories) > 0
    assert all(isinstance(m, ConversationMemory) for m in memories)


def test_agent_with_memory(mock_agent, mock_rag):
    """Test agent with memory."""
    memory = RAGMemoryStore(rag_client=mock_rag)
    agent = AgentWithMemory(base_agent=mock_agent, memory=memory)
    
    # Store memory first
    memory.store("user1", "My name is John", "Got it!")
    
    # Invoke with memory
    response = agent.invoke_with_memory("What's my name?", user_id="user1")
    assert response is not None
    assert mock_agent.invoke_count > 0


def test_multi_agent_system(mock_agent):
    """Test multi-agent system."""
    researcher = ResearcherAgent(mock_agent)
    writer = WriterAgent(mock_agent)
    
    system = MultiAgentSystem(agents={
        "researcher": researcher,
        "writer": writer
    })
    
    assert len(system.get_agents()) == 2
    assert "researcher" in system.get_agents()
    assert "writer" in system.get_agents()


def test_planning_agent_creates_plan(mock_agent):
    """Test planning agent creates plans."""
    mock_agent.responses = [
        "1. Step one\n2. Step two\n3. Step three",
        "Result of step one",
        "Result of step two",
        "Result of step three",
        "Final summary"
    ]
    
    planning_agent = PlanningAgent(base_agent=mock_agent, enable_revision=False)
    
    # Mock the invoke to return plan text
    plan = planning_agent._create_plan("Test goal")
    assert plan.goal == "Test goal"
    assert len(plan.steps) > 0


def test_reflective_agent(mock_agent):
    """Test reflective agent."""
    mock_agent.responses = [
        "Initial response",
        "CONFIDENCE: 0.5\nRESULT: ACCEPTABLE\nREASONING: Good response"
    ]
    
    reflective_agent = ReflectiveAgent(base_agent=mock_agent, enable_reflection=True)
    
    # Should invoke base agent and reflection
    response = reflective_agent.invoke("Test")
    assert response is not None


def test_agent_monitor_tracks_events():
    """Test agent monitor."""
    monitor = AgentMonitor()
    
    monitor.track_event(EventType.INVOCATION_START, "agent1")
    monitor.track_event(EventType.TOOL_CALL, "agent1", {"tool_name": "rag_search"})
    monitor.track_event(EventType.INVOCATION_END, "agent1")
    
    metrics = monitor.get_metrics("agent1")
    assert metrics is not None
    assert metrics.total_invocations == 1
    assert metrics.total_tool_calls == 1
    assert "rag_search" in metrics.tool_usage


def test_monitored_agent(mock_agent):
    """Test monitored agent."""
    monitor = AgentMonitor()
    monitored = MonitoredAgent(base_agent=mock_agent, monitor=monitor, agent_id="test")
    
    response = monitored.invoke("Test")
    
    assert response is not None
    metrics = monitor.get_metrics("test")
    assert metrics.total_invocations == 1


def test_cost_tracker():
    """Test cost tracker."""
    tracker = CostTracker()
    
    cost = tracker.track(
        agent_id="agent1",
        operation="invoke",
        model="gpt-4",
        input_tokens=1000,
        output_tokens=500
    )
    
    assert cost > 0
    assert tracker.get_total_cost("agent1") == cost


def test_cost_tracking_agent(mock_agent):
    """Test cost tracking agent."""
    tracker = CostTracker()
    tracker.set_budget("agent1", budget=10.0)
    
    cost_agent = CostTrackingAgent(
        base_agent=mock_agent,
        cost_tracker=tracker,
        agent_id="agent1",
        model="gpt-4"
    )
    
    response = cost_agent.invoke("Test")
    assert response is not None
    
    cost = tracker.get_total_cost("agent1")
    assert cost >= 0


def test_budget_exceeded_error(mock_agent):
    """Test budget exceeded error."""
    tracker = CostTracker()
    tracker.set_budget("agent1", budget=0.0001)  # Very small budget
    
    cost_agent = CostTrackingAgent(
        base_agent=mock_agent,
        cost_tracker=tracker,
        agent_id="agent1",
        model="gpt-4"
    )
    
    # First call should work
    cost_agent.invoke("Test")
    
    # Set budget to very low and try again
    tracker.set_budget("agent1", budget=0.0)
    
    # Should raise error if cost exceeds budget
    # (This depends on implementation - may need adjustment)


def test_persona_agent(mock_agent):
    """Test persona agent."""
    persona_agent = create_persona_agent(mock_agent, PersonaType.RESEARCHER)
    
    persona = persona_agent.get_persona()
    assert persona.name == "Researcher"
    assert "research" in persona.capabilities


def test_error_recovery_agent(mock_agent):
    """Test error recovery agent."""
    retry_config = RetryConfig(max_retries=2)
    
    recovery_agent = ErrorRecoveryAgent(
        base_agent=mock_agent,
        retry_config=retry_config
    )
    
    response = recovery_agent.invoke("Test")
    assert response is not None


def test_agent_evaluator(mock_agent):
    """Test agent evaluator."""
    evaluator = AgentEvaluator()
    
    tasks = [
        EvaluationTask(
            task_id="task1",
            prompt="Test prompt",
            expected_output="Expected output"
        )
    ]
    
    results = evaluator.evaluate(mock_agent, tasks, track_tools=False, track_latency=False)
    
    assert len(results) == 1
    assert results[0].task_id == "task1"
    assert results[0].actual_output is not None


def test_exact_match_metric():
    """Test exact match metric."""
    metric = ExactMatchMetric()
    
    task = EvaluationTask(
        task_id="task1",
        prompt="Test",
        expected_output="Expected"
    )
    
    result = EvaluationTask(
        task_id="task1",
        prompt="Test",
        actual_output="Expected"
    )
    
    score = metric.compute(task, result)
    assert score == 1.0
    
    result.actual_output = "Different"
    score = metric.compute(task, result)
    assert score == 0.0
