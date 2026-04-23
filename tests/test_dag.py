import pytest
from llmx.dag import DagNode, Dag, validate_dag, format_dag_preview, DagValidationError


def test_valid_dag():
    dag = Dag(
        id="dag_001",
        prompt="test prompt",
        nodes=[
            DagNode(id="n1", task="do thing", category="speed", depends_on=[]),
            DagNode(id="n2", task="use {n1}", category="code", depends_on=["n1"]),
        ],
    )
    validate_dag(dag)  # should not raise


def test_circular_dependency_raises():
    dag = Dag(
        id="dag_002",
        prompt="test",
        nodes=[
            DagNode(id="n1", task="a", category="speed", depends_on=["n2"]),
            DagNode(id="n2", task="b", category="speed", depends_on=["n1"]),
        ],
    )
    with pytest.raises(DagValidationError, match="Circular"):
        validate_dag(dag)


def test_missing_dependency_raises():
    dag = Dag(
        id="dag_003",
        prompt="test",
        nodes=[
            DagNode(id="n1", task="a", category="speed", depends_on=["n99"]),
        ],
    )
    with pytest.raises(DagValidationError, match="n99"):
        validate_dag(dag)


def test_invalid_category_raises():
    with pytest.raises(ValueError):
        DagNode(id="n1", task="a", category="invalid_cat", depends_on=[])


def test_duplicate_node_ids_raises():
    dag = Dag(
        id="dag_004",
        prompt="test",
        nodes=[
            DagNode(id="n1", task="a", category="speed", depends_on=[]),
            DagNode(id="n1", task="b", category="speed", depends_on=[]),
        ],
    )
    with pytest.raises(DagValidationError, match="Duplicate"):
        validate_dag(dag)


def test_format_dag_preview():
    dag = Dag(
        id="dag_001",
        prompt="test",
        nodes=[
            DagNode(id="n1", task="do thing", category="speed", depends_on=[]),
            DagNode(id="n2", task="use n1 result", category="code", depends_on=["n1"]),
        ],
    )
    preview = format_dag_preview(dag)
    assert "n1" in preview
    assert "n2" in preview
    assert "speed" in preview
    assert "code" in preview


def test_compute_waves():
    from llmx.dag import compute_waves
    dag = Dag(
        id="dag_005",
        prompt="test",
        nodes=[
            DagNode(id="n1", task="a", category="speed", depends_on=[]),
            DagNode(id="n2", task="b", category="speed", depends_on=[]),
            DagNode(id="n3", task="c", category="code", depends_on=["n1", "n2"]),
        ],
    )
    waves = compute_waves(dag)
    assert waves == [["n1", "n2"], ["n3"]]
