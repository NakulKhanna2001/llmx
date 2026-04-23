from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator


class Category(str, Enum):
    SPEED = "speed"
    CODE = "code"
    RESEARCH = "research"
    LARGE_CONTEXT = "large_context"
    REASONING = "reasoning"
    CREATIVE = "creative"


class DagValidationError(Exception):
    pass


class DagNode(BaseModel):
    id: str
    task: str
    category: Category
    depends_on: list[str] = []


class Dag(BaseModel):
    id: str
    prompt: str
    nodes: list[DagNode]


def validate_dag(dag: Dag) -> None:
    node_ids = [n.id for n in dag.nodes]

    # Check duplicate IDs
    if len(node_ids) != len(set(node_ids)):
        seen = set()
        for nid in node_ids:
            if nid in seen:
                raise DagValidationError(f"Duplicate node id: {nid}")
            seen.add(nid)

    id_set = set(node_ids)

    # Check all dependencies reference existing nodes
    for node in dag.nodes:
        for dep in node.depends_on:
            if dep not in id_set:
                raise DagValidationError(f"Node '{node.id}' depends on unknown node '{dep}'")

    # Check for circular dependencies via topological sort
    in_degree = {n.id: len(n.depends_on) for n in dag.nodes}
    deps_map = {n.id: set(n.depends_on) for n in dag.nodes}
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    visited = 0

    while queue:
        current = queue.pop(0)
        visited += 1
        for node in dag.nodes:
            if current in deps_map[node.id]:
                deps_map[node.id].discard(current)
                in_degree[node.id] -= 1
                if in_degree[node.id] == 0:
                    queue.append(node.id)

    if visited != len(dag.nodes):
        raise DagValidationError("Circular dependency detected in DAG")


def compute_waves(dag: Dag) -> list[list[str]]:
    validate_dag(dag)
    remaining = {n.id: set(n.depends_on) for n in dag.nodes}
    waves = []

    while remaining:
        wave = [nid for nid, deps in remaining.items() if len(deps) == 0]
        if not wave:
            raise DagValidationError("Circular dependency detected in DAG")
        wave.sort()  # deterministic ordering
        waves.append(wave)
        for nid in wave:
            del remaining[nid]
        for deps in remaining.values():
            deps -= set(wave)

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
