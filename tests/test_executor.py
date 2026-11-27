from unittest.mock import MagicMock

from llmx.dag import Dag, DagNode
from llmx.executor import DagExecutor
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


def test_dependency_inputs_expand_once_and_only_from_declared_parents():
    dag = Dag(id="run", prompt="test", nodes=[
        DagNode(id="left", task="left input", category="speed"),
        DagNode(id="right", task="right input", category="reasoning"),
        DagNode(id="merge", task='Join {left} + {right}; keep {"a":1}', category="code",
                depends_on=["right", "left"]),
    ])
    p1 = make_mock_provider("groq", "{right}")
    p2 = make_mock_provider("openai", "correct")
    p3 = make_mock_provider("openrouter", "merged")
    ps = {"groq": p1, "openai": p2, "openrouter": p3}
    DagExecutor(dag, ps, list(ps)).execute()
    assert p3.complete.call_args.args == ('Join {right} + correct; keep {"a":1}',)


def test_provider_exception_and_failed_dependency_preserve_independent_results():
    dag = Dag(id="run", prompt="test", nodes=[
        DagNode(id="bad", task="break", category="speed"),
        DagNode(id="child", task="do not call", category="code", depends_on=["bad"]),
        DagNode(id="ok", task="hello", category="speed"),
    ])
    provider = make_mock_provider("groq", "ok")
    healthy = provider.complete.return_value
    def complete(prompt, model=None):
        if prompt == "hello":
            return healthy
        raise TimeoutError("unavailable")

    provider.complete.side_effect = complete
    result = DagExecutor(dag, {"groq": provider}, ["groq"], max_workers=1).execute()
    assert result.results["ok"].success
    assert result.results["bad"].fallback_exhausted
    assert result.results["child"].blocked_by == ["bad"]
    assert not result.results["child"].fallback_exhausted
    assert provider.complete.call_count == 2


def test_ready_nodes_do_not_wait_for_unrelated_branch():
    from threading import Event
    signal = Event()
    dag = Dag(id="run", prompt="test", nodes=[
        DagNode(id="slow", task="wait for grandchild", category="speed"),
        DagNode(id="root", task="root", category="speed"),
        DagNode(id="child", task="set signal", category="code", depends_on=["root"]),
    ])
    provider = make_mock_provider("groq", "")
    def complete(prompt, model=None):
        if prompt == "wait for grandchild":
            assert signal.wait(timeout=2)
        elif prompt == "set signal":
            signal.set()
        return ProviderResult(output="ok", success=True, provider="groq", model="test")
    provider.complete.side_effect = complete
    result = DagExecutor(dag, {"groq": provider}, ["groq"], max_workers=2).execute()
    assert all(n.success for n in result.results.values())


def test_consumer_first_with_one_worker_terminates():
    import subprocess
    import sys
    script = """
from llmx.dag import Dag, DagNode
from llmx.executor import DagExecutor
dag = Dag(id='d', prompt='p', nodes=[
    DagNode(id='child', task='{root}', category='speed', depends_on=['root']),
    DagNode(id='root', task='obtain input', category='speed')])
result = DagExecutor(dag, {}, [], max_workers=1).execute()
assert result.results['child'].blocked_by == ['root']
"""
    subprocess.run([sys.executable, "-c", script], timeout=4, check=True)


def test_provider_exception_advances_to_next_model():
    primary = make_mock_provider("groq", "unused")
    primary.complete.side_effect = TimeoutError()
    fallback = make_mock_provider("openai", "recovered")
    dag = Dag(id="d", prompt="p", nodes=[DagNode(id="one", task="ask", category="speed")])
    providers = {"groq": primary, "openai": fallback}
    result = DagExecutor(dag, providers, ["missing", "groq", "openai"]).execute()
    assert result.results["one"].output == "recovered"
    assert result.retries["one"] == ["groq:TimeoutError"]
