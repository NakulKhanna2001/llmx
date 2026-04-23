import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Event

from .dag import Dag, DagNode, compute_waves
from .fallback import get_provider_chain, get_model_for_provider
from .providers.base import BaseProvider, ProviderResult


@dataclass
class NodeResult:
    output: str
    success: bool
    provider: str
    model: str
    latency_ms: int = 0
    error_message: str = ""
    fallback_exhausted: bool = False


@dataclass
class ExecutionResult:
    results: dict[str, NodeResult] = field(default_factory=dict)
    execution_time_ms: int = 0
    retries: dict[str, list[str]] = field(default_factory=dict)


class DagExecutor:
    def __init__(
        self,
        dag: Dag,
        providers: dict[str, BaseProvider],
        available_providers: list[str],
        max_workers: int = 8,
    ):
        self.dag = dag
        self.providers = providers
        self.available_providers = available_providers
        self.max_workers = max_workers
        self.node_outputs: dict[str, str] = {}
        self.node_events: dict[str, Event] = {n.id: Event() for n in dag.nodes}

    def _substitute_placeholders(self, task: str) -> str:
        def replacer(match: re.Match) -> str:
            node_id = match.group(1)
            return self.node_outputs.get(node_id, f"{{{node_id}}}")
        return re.sub(r"\{(n\d+)\}", replacer, task)

    def _execute_node(self, node: DagNode) -> NodeResult:
        # Wait for dependencies
        for dep_id in node.depends_on:
            self.node_events[dep_id].wait()

        # Substitute placeholders
        prompt = self._substitute_placeholders(node.task)

        # Get provider chain for this category
        chain = get_provider_chain(node.category.value, self.available_providers)
        retries = []

        for provider_name in chain:
            provider = self.providers[provider_name]
            model = get_model_for_provider(provider_name, node.category.value)

            result = provider.complete(prompt, model=model)

            if result.success:
                self.node_outputs[node.id] = result.output
                self.node_events[node.id].set()
                if retries:
                    self.retries[node.id] = retries
                return NodeResult(
                    output=result.output,
                    success=True,
                    provider=result.provider,
                    model=result.model,
                    latency_ms=result.latency_ms,
                )
            else:
                retries.append(f"{provider_name}:{result.error_code or 'error'}")

        # All providers failed
        self.node_outputs[node.id] = ""
        self.node_events[node.id].set()
        if retries:
            self.retries[node.id] = retries
        return NodeResult(
            output="",
            success=False,
            provider="none",
            model="none",
            error_message="All providers failed",
            fallback_exhausted=True,
        )

    def execute(self) -> ExecutionResult:
        self.retries: dict[str, list[str]] = {}
        start = time.monotonic()
        results: dict[str, NodeResult] = {}
        node_map = {n.id: n for n in self.dag.nodes}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for node in self.dag.nodes:
                future = pool.submit(self._execute_node, node)
                futures[future] = node.id

            for future in as_completed(futures):
                node_id = futures[future]
                results[node_id] = future.result()

        execution_time = int((time.monotonic() - start) * 1000)
        return ExecutionResult(
            results=results,
            execution_time_ms=execution_time,
            retries=self.retries,
        )
