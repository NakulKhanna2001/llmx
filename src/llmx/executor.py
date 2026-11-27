import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from .dag import PLACEHOLDER, Dag, DagNode, compute_waves, prepare_graph
from .fallback import get_model_for_provider, get_provider_chain
from .providers.base import BaseProvider


@dataclass
class NodeResult:
    output: str
    success: bool
    provider: str
    model: str
    latency_ms: int = 0
    error_message: str = ""
    fallback_exhausted: bool = False
    blocked_by: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    results: dict[str, NodeResult] = field(default_factory=dict)
    execution_time_ms: int = 0
    retries: dict[str, list[str]] = field(default_factory=dict)
    attempted: dict[str, set[str]] = field(default_factory=dict)


class DagExecutor:
    def __init__(self, dag: Dag, providers: dict[str, BaseProvider],
                 available_providers: list[str], max_workers: int = 8):
        self.dag = dag
        self.nodes = {node.id: node for node in dag.nodes}
        self.providers = providers
        self.available_providers = [name for name in available_providers if name in providers]
        self.max_workers = max_workers

    def _run_node(self, node: DagNode, parents: dict[str, NodeResult],
                  skip_providers: list[str] | None = None,
                  replacement_output: str | None = None) -> tuple[NodeResult, list[str]]:
        blocked = [parent for parent, result in parents.items() if not result.success]
        if blocked:
            return NodeResult("", False, "none", "none", blocked_by=blocked,
                              error_message="Unresolved dependencies: " + ", ".join(blocked)), []
        if replacement_output is not None:
            return NodeResult(replacement_output, True, "client", "provided"), []

        def substitute(match):
            parent = parents.get(match.group(1))
            return parent.output if parent is not None else match.group(0)

        prompt = PLACEHOLDER.sub(substitute, node.task)
        chain = get_provider_chain(node.category.value, self.available_providers, skip_providers)
        failures = []
        for name in chain:
            provider = self.providers[name]
            model = get_model_for_provider(name, node.category.value)
            try:
                if node.attachments:
                    result = provider.complete_with_attachments(prompt, node.attachments, model=model)
                else:
                    result = provider.complete(prompt, model=model)
            except Exception as exc:
                # A broken provider must not prevent fallback or strand other nodes.
                failures.append(f"{name}:{type(exc).__name__}")
                continue
            if result.success:
                return NodeResult(result.output, True, result.provider, result.model,
                                  result.latency_ms), failures
            failures.append(f"{name}:{result.error_code or 'error'}")

        return NodeResult("", False, "none", "none", fallback_exhausted=True,
                          error_message="All providers failed" if chain else "No eligible providers"), failures

    def execute(self) -> ExecutionResult:
        start = time.monotonic()
        graph = prepare_graph(self.dag)
        run = ExecutionResult()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            pending = {}
            while graph.is_active():
                for node_id in graph.get_ready():
                    node = self.nodes[node_id]
                    parents = {dep: run.results[dep] for dep in node.depends_on}
                    pending[pool.submit(self._run_node, node, parents)] = node_id
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    node_id = pending.pop(future)
                    result, failures = future.result()
                    run.results[node_id] = result
                    self._record_attempts(run, node_id, result, failures)
                    graph.done(node_id)
        run.execution_time_ms = int((time.monotonic() - start) * 1000)
        return run

    @staticmethod
    def _record_attempts(run: ExecutionResult, node_id: str, result: NodeResult,
                         failures: list[str]) -> None:
        attempted = run.attempted.setdefault(node_id, set())
        if failures:
            run.retries.setdefault(node_id, []).extend(failures)
            attempted.update(failure.split(":", 1)[0] for failure in failures)
        if result.success and result.provider != "client":
            attempted.add(result.provider)

    def retry_node(self, run: ExecutionResult, node_id: str, skip_providers: list[str] | None,
                   replacement_output: str | None) -> tuple[NodeResult, list[str]]:
        node = self.nodes[node_id]
        parents = {dep: run.results[dep] for dep in node.depends_on}
        if skip_providers is None:
            skip_providers = list(run.attempted.get(node_id, ()))
        result, failures = self._run_node(node, parents, skip_providers, replacement_output)
        self._record_attempts(run, node_id, result, failures)
        previous = run.results[node_id]
        if not result.success:
            if not previous.success:
                run.results[node_id] = result
            return result, []

        run.results[node_id] = result
        if previous.success and previous.output == result.output:
            return result, []

        # A successful upstream correction makes all previous downstream outputs stale.
        # Leave explicit retries to the caller (they may incur provider charges).
        changed = {node_id}
        invalidated = []
        for wave in compute_waves(self.dag):
            for child_id in wave:
                child = self.nodes[child_id]
                if changed.isdisjoint(child.depends_on):
                    continue
                changed.add(child_id)
                invalidated.append(child_id)
                blocked = [dep for dep in child.depends_on if not run.results[dep].success]
                run.results[child_id] = NodeResult("", False, "none", "none", blocked_by=blocked,
                                                   error_message="An input changed; retry this node")
                run.attempted.pop(child_id, None)
                run.retries.pop(child_id, None)
        return result, invalidated
