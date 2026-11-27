import re
from enum import Enum
from graphlib import CycleError, TopologicalSorter
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from .attachments import Attachment


class Category(str, Enum):
    SPEED = "speed"
    CODE = "code"
    RESEARCH = "research"
    LARGE_CONTEXT = "large_context"
    REASONING = "reasoning"
    CREATIVE = "creative"


class DagValidationError(Exception):
    pass


def nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("A nonblank value is required")
    return value


NonblankText = Annotated[str, AfterValidator(nonblank)]


class DagNode(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    id: NonblankText
    task: NonblankText
    attachments: list[Attachment] = Field(default_factory=list, max_length=4)
    category: Category
    depends_on: list[str] = Field(default_factory=list)


class Dag(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    id: NonblankText
    prompt: NonblankText
    nodes: list[DagNode] = Field(min_length=1)


# Replace declared dependency names only. Other braces, e.g. JSON examples, are literal.
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def prepare_graph(dag: Dag) -> TopologicalSorter:
    nodes = {}
    for node in dag.nodes:
        if node.id in nodes:
            raise DagValidationError(f"Duplicate node id: {node.id}")
        nodes[node.id] = node

    graph = TopologicalSorter()
    for node in dag.nodes:
        dependencies = set(node.depends_on)
        if len(dependencies) != len(node.depends_on):
            raise DagValidationError(f"Duplicate dependency in node '{node.id}'")
        for dep in node.depends_on:
            if dep not in nodes:
                raise DagValidationError(f"Node '{node.id}' depends on unknown node '{dep}'")
        for ref in PLACEHOLDER.findall(node.task):
            if (ref in nodes or re.fullmatch(r"n\d+", ref)) and ref not in dependencies:
                raise DagValidationError(
                    f"Node '{node.id}' uses placeholder '{{{ref}}}' without a declared dependency"
                )
        graph.add(node.id, *node.depends_on)

    try:
        graph.prepare()
    except CycleError as exc:
        raise DagValidationError("Circular dependency detected in DAG") from exc
    return graph


def validate_dag(dag: Dag) -> None:
    prepare_graph(dag)


def compute_waves(dag: Dag) -> list[list[str]]:
    graph = prepare_graph(dag)
    waves = []
    while graph.is_active():
        wave = sorted(graph.get_ready())
        waves.append(wave)
        graph.done(*wave)
    return waves


def format_dag_preview(dag: Dag) -> str:
    waves = compute_waves(dag)
    node_map = {n.id: n for n in dag.nodes}
    lines = []
    lines.append(f"DAG: {dag.id} — {len(dag.nodes)} nodes, {len(waves)} waves")
    lines.append("")
    for i, wave in enumerate(waves):
        lines.append(f"  Wave {i + 1} ({'parallel' if len(wave) > 1 else 'sequential'}):")
        for nid in wave:
            node = node_map[nid]
            task_short = node.task[:60] + "..." if len(node.task) > 60 else node.task
            deps = ", ".join(node.depends_on) if node.depends_on else "none"
            lines.append(f"    [{nid}] {task_short}")
            lines.append(f"         category={node.category.value}  depends_on={deps}")
    return "\n".join(lines)
