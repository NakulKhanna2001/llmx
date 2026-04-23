"""
Full round-trip test: build a DAG, execute it with mocked providers, verify results.
"""
from unittest.mock import MagicMock

from llmx.dag import Dag, DagNode, validate_dag, format_dag_preview, compute_waves
from llmx.executor import DagExecutor
from llmx.providers.base import ProviderResult


def make_provider(name: str, response: str):
    p = MagicMock()
    p.name = name
    p.default_model = "test-model"
    p.complete.return_value = ProviderResult(
        output=response, success=True, provider=name, model="test-model", latency_ms=50,
    )
    return p


def test_full_dag_roundtrip():
    # 1. Build a DAG like Claude would
    dag = Dag(
        id="integration_001",
        prompt="Compare Python and Rust for CLI tools",
        nodes=[
            DagNode(id="n1", task="List Python advantages for CLI development", category="speed", depends_on=[]),
            DagNode(id="n2", task="List Rust advantages for CLI development", category="speed", depends_on=[]),
            DagNode(id="n3", task="Find popular CLI tools built in Python vs Rust", category="research", depends_on=[]),
            DagNode(id="n4", task="Given: Python({n1}), Rust({n2}), examples({n3}): create comparison", category="reasoning", depends_on=["n1", "n2", "n3"]),
        ],
    )

    # 2. Validate
    validate_dag(dag)

    # 3. Preview
    preview = format_dag_preview(dag)
    assert "4 nodes" in preview
    assert "2 waves" in preview

    # 4. Compute waves
    waves = compute_waves(dag)
    assert len(waves) == 2
    assert set(waves[0]) == {"n1", "n2", "n3"}
    assert waves[1] == ["n4"]

    # 5. Execute with mocked providers
    providers = {
        "groq": make_provider("groq", "Python is great for rapid CLI dev"),
        "openai": make_provider("openai", "Comprehensive comparison table"),
        "openrouter": make_provider("openrouter", "Click, Typer, Clap, StructOpt"),
    }
    executor = DagExecutor(
        dag, providers,
        available_providers=["groq", "openai", "openrouter"],
    )
    result = executor.execute()

    # 6. Verify all nodes completed
    assert len(result.results) == 4
    for node_id in ["n1", "n2", "n3", "n4"]:
        assert result.results[node_id].success is True
        assert result.results[node_id].output != ""

    # 7. Verify n4 got substituted prompt
    n4_call = providers["openai"].complete.call_args
    prompt = n4_call[0][0]
    assert "Python is great for rapid CLI dev" in prompt
    assert "Click, Typer, Clap, StructOpt" in prompt

    # 8. Verify execution metadata
    assert result.execution_time_ms >= 0
