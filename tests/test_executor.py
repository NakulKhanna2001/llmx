import time
from unittest.mock import MagicMock

from llmx.dag import Dag, DagNode
from llmx.executor import DagExecutor, ExecutionResult
from llmx.providers.base import ProviderResult


def make_mock_provider(name: str, response: str):
    provider = MagicMock()
    provider.name = name
    provider.default_model = "test-model"
    provider.complete.return_value = ProviderResult(
        output=response,
        success=True,
        provider=name,
        model="test-model",
        latency_ms=100,
    )
    return provider


def test_executor_simple_dag():
    dag = Dag(
        id="test",
        prompt="test",
        nodes=[
            DagNode(id="n1", task="do thing", category="speed", depends_on=[]),
        ],
    )
    providers = {"groq": make_mock_provider("groq", "result1")}
    executor = DagExecutor(dag, providers, available_providers=["groq"])
    result = executor.execute()
    assert "n1" in result.results
    assert result.results["n1"].output == "result1"


def test_executor_parallel_nodes():
    dag = Dag(
        id="test",
        prompt="test",
        nodes=[
            DagNode(id="n1", task="task a", category="speed", depends_on=[]),
            DagNode(id="n2", task="task b", category="speed", depends_on=[]),
        ],
    )
    providers = {"groq": make_mock_provider("groq", "result")}
    executor = DagExecutor(dag, providers, available_providers=["groq"])
    result = executor.execute()
    assert "n1" in result.results
    assert "n2" in result.results


def test_executor_dependency_substitution():
    dag = Dag(
        id="test",
        prompt="test",
        nodes=[
            DagNode(id="n1", task="first task", category="speed", depends_on=[]),
            DagNode(id="n2", task="use {n1} here", category="reasoning", depends_on=["n1"]),
        ],
    )
    groq = make_mock_provider("groq", "groq_output")
    openai = make_mock_provider("openai", "openai_output")
    providers = {"groq": groq, "openai": openai}
    executor = DagExecutor(dag, providers, available_providers=["groq", "openai"])
    result = executor.execute()

    # n2 should have been called with the substituted prompt
    call_args = openai.complete.call_args
    assert "groq_output" in call_args[0][0]


def test_executor_fallback_on_failure():
    dag = Dag(
        id="test",
        prompt="test",
        nodes=[
            DagNode(id="n1", task="do thing", category="speed", depends_on=[]),
        ],
    )
    failing_groq = MagicMock()
    failing_groq.name = "groq"
    failing_groq.default_model = "test"
    failing_groq.complete.return_value = ProviderResult(
        output="", success=False, provider="groq", model="test",
        latency_ms=100, error_code=429, error_message="rate limited",
    )
    openai = make_mock_provider("openai", "fallback_result")
    providers = {"groq": failing_groq, "openai": openai}
    executor = DagExecutor(dag, providers, available_providers=["groq", "openai"])
    result = executor.execute()
    assert result.results["n1"].output == "fallback_result"
    assert result.results["n1"].provider == "openai"
    assert "n1" in result.retries


def test_executor_all_providers_fail():
    dag = Dag(
        id="test",
        prompt="test",
        nodes=[
            DagNode(id="n1", task="do thing", category="speed", depends_on=[]),
        ],
    )
    failing = MagicMock()
    failing.name = "groq"
    failing.default_model = "test"
    failing.complete.return_value = ProviderResult(
        output="", success=False, provider="groq", model="test",
        latency_ms=100, error_code=500, error_message="error",
    )
    providers = {"groq": failing}
    executor = DagExecutor(dag, providers, available_providers=["groq"])
    result = executor.execute()
    assert result.results["n1"].success is False
    assert result.results["n1"].fallback_exhausted is True
