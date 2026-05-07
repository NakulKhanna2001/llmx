# Tavern Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python CLI (`tavern`) that orchestrates complex prompts across 9 free LLM providers using Claude as Guild Master for decomposition and synthesis.

**Architecture:** Guild metaphor — Courtyard (shared context), Bulletin Board (quest DAG with curated context), Guild Master (Claude via Anthropic SDK), Adventurers (9 LLM providers). Two Claude API calls: decompose prompt into DAG, then synthesize results. Category-based provider routing with fallback chains.

**Tech Stack:** Python 3.11+, httpx, rich, pydantic, pyyaml, anthropic SDK, pytest + respx

---

## File Structure

```
~/tavern/
├── pyproject.toml
├── .gitignore
├── README.md
├── src/tavern/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── courtyard.py
│   ├── guildmaster.py
│   ├── dag.py
│   ├── executor.py
│   ├── fallback.py
│   ├── config.py
│   ├── wizard.py
│   └── providers/
│       ├── __init__.py
│       ├── base.py
│       ├── groq.py
│       ├── cerebras.py
│       ├── sambanova.py
│       ├── gemini.py
│       ├── openrouter.py
│       ├── mistral.py
│       ├── github_models.py
│       ├── zhipu.py
│       └── ollama.py
└── tests/
    ├── test_courtyard.py
    ├── test_dag.py
    ├── test_fallback.py
    ├── test_config.py
    ├── test_providers.py
    ├── test_executor.py
    ├── test_guildmaster.py
    ├── test_wizard.py
    ├── test_cli.py
    └── test_integration.py
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `~/tavern/pyproject.toml`
- Create: `~/tavern/.gitignore`
- Create: `~/tavern/src/tavern/__init__.py`
- Create: `~/tavern/src/tavern/__main__.py`

- [ ] **Step 1: Create project directory and initialize git**

```bash
mkdir -p ~/tavern/src/tavern/providers ~/tavern/tests
cd ~/tavern
git init
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
*.pyo
.venv/
*.egg-info/
dist/
build/
.pytest_cache/
.mypy_cache/
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tavern"
version = "0.1.0"
description = "Multi-LLM orchestrator CLI — Guild Master decomposes prompts, Adventurers execute"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "rich>=13.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "anthropic>=0.40",
]

[project.scripts]
tavern = "tavern.cli:main"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "respx>=0.21",
]

[tool.hatch.build.targets.wheel]
packages = ["src/tavern"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create `src/tavern/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 5: Create `src/tavern/__main__.py`**

```python
from tavern.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Create venv and install in editable mode**

```bash
cd ~/tavern
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 7: Verify import works**

```bash
cd ~/tavern
.venv/bin/python -c "import tavern; print(tavern.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore src/tavern/__init__.py src/tavern/__main__.py
git commit -m "feat: project scaffolding — pyproject.toml, venv, package init"
```

Note: `cli.py` doesn't exist yet so `tavern` entry point won't work — that's fine, we'll create it in Task 14.

---

### Task 2: Provider Base Class

**Files:**
- Create: `src/tavern/providers/__init__.py` (empty for now)
- Create: `src/tavern/providers/base.py`
- Create: `tests/test_providers.py`

Adapted from `~/llmx/src/llmx/providers/base.py` — same pattern.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers.py
from tavern.providers.base import BaseProvider, ProviderResult


class TestProviderResult:
    def test_success_result(self):
        result = ProviderResult(
            output="hello",
            success=True,
            provider="test",
            model="test-model",
            latency_ms=100,
        )
        assert result.output == "hello"
        assert result.success is True
        assert result.provider == "test"
        assert result.model == "test-model"
        assert result.latency_ms == 100
        assert result.error_code is None
        assert result.error_message == ""

    def test_failure_result(self):
        result = ProviderResult(
            output="",
            success=False,
            provider="test",
            model="test-model",
            error_code=429,
            error_message="Rate limited",
        )
        assert result.success is False
        assert result.error_code == 429
        assert result.error_message == "Rate limited"

    def test_base_provider_is_abstract(self):
        import pytest
        with pytest.raises(TypeError):
            BaseProvider()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_providers.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create `src/tavern/providers/__init__.py`** (empty file)

```python
```

- [ ] **Step 4: Create `src/tavern/providers/base.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResult:
    output: str
    success: bool
    provider: str
    model: str
    latency_ms: int = 0
    error_code: int | None = None
    error_message: str = ""


class BaseProvider(ABC):
    name: str
    default_model: str

    @abstractmethod
    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        pass

    @abstractmethod
    def validate(self) -> bool:
        pass
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_providers.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/tavern/providers/__init__.py src/tavern/providers/base.py tests/test_providers.py
git commit -m "feat: provider base class — ProviderResult dataclass and BaseProvider ABC"
```

---

### Task 3: Config Module

**Files:**
- Create: `src/tavern/config.py`
- Create: `tests/test_config.py`

Adapted from `~/llmx/src/llmx/config.py` — adds `guild_master` section, new config path `~/.tavern/config.yaml`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import tempfile
from pathlib import Path

from tavern.config import TavernConfig, load_config, save_config


class TestTavernConfig:
    def test_default_config(self):
        config = TavernConfig()
        assert config.guild_master == {}
        assert config.adventurers == {}

    def test_get_available_adventurers_empty(self):
        config = TavernConfig()
        assert config.get_available_adventurers() == []

    def test_get_available_adventurers_with_validated(self):
        config = TavernConfig(
            adventurers={
                "groq": {"api_key": "gsk_test", "validated": True},
                "cerebras": {"api_key": "csk_test", "validated": False},
                "ollama": {"available": True},
            }
        )
        available = config.get_available_adventurers()
        assert "groq" in available
        assert "cerebras" not in available
        assert "ollama" in available

    def test_get_anthropic_key_from_config(self):
        config = TavernConfig(guild_master={"api_key": "sk-ant-test"})
        assert config.get_anthropic_key() == "sk-ant-test"

    def test_get_anthropic_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        config = TavernConfig()
        assert config.get_anthropic_key() == "sk-ant-env"

    def test_env_key_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        config = TavernConfig(guild_master={"api_key": "sk-ant-config"})
        assert config.get_anthropic_key() == "sk-ant-env"


class TestConfigIO:
    def test_save_and_load(self):
        config = TavernConfig(
            guild_master={"api_key": "sk-ant-test", "model": "claude-sonnet-4-5-20250514"},
            adventurers={
                "groq": {"api_key": "gsk_test", "validated": True},
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            save_config(config, path)
            loaded = load_config(path)
            assert loaded.guild_master["api_key"] == "sk-ant-test"
            assert loaded.adventurers["groq"]["api_key"] == "gsk_test"

    def test_load_missing_file(self):
        config = load_config(Path("/nonexistent/config.yaml"))
        assert config.guild_master == {}
        assert config.adventurers == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create `src/tavern/config.py`**

```python
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path.home() / ".tavern" / "config.yaml"


@dataclass
class TavernConfig:
    guild_master: dict[str, Any] = field(default_factory=dict)
    adventurers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_available_adventurers(self) -> list[str]:
        available = []
        for name, info in self.adventurers.items():
            if name == "ollama":
                if info.get("available", False):
                    available.append(name)
            elif info.get("validated", False) and info.get("api_key"):
                available.append(name)
        return available

    def get_anthropic_key(self) -> str | None:
        env_key = os.environ.get("ANTHROPIC_API_KEY")
        if env_key:
            return env_key
        return self.guild_master.get("api_key")

    def get_guild_master_model(self) -> str:
        return self.guild_master.get("model", "claude-sonnet-4-5-20250514")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> TavernConfig:
    if not path.exists():
        return TavernConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return TavernConfig(
        guild_master=data.get("guild_master", {}),
        adventurers=data.get("adventurers", {}),
    )


def save_config(config: TavernConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "guild_master": config.guild_master,
        "adventurers": config.adventurers,
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/tavern/config.py tests/test_config.py
git commit -m "feat: config module — TavernConfig with guild_master + adventurers, env var support"
```

---

### Task 4: DAG Schema (Quest)

**Files:**
- Create: `src/tavern/dag.py`
- Create: `tests/test_dag.py`

Adapted from `~/llmx/src/llmx/dag.py` — adds `context` and `priority` fields, renames node→quest and `nX`→`qX`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dag.py
import pytest

from tavern.dag import (
    Category,
    Priority,
    Quest,
    QuestDAG,
    QuestValidationError,
    validate_dag,
    compute_waves,
    format_quest_board,
)


class TestQuestModel:
    def test_create_quest(self):
        q = Quest(id="q1", task="Do something", context="some context", category=Category.SPEED)
        assert q.id == "q1"
        assert q.context == "some context"
        assert q.priority == Priority.NORMAL
        assert q.depends_on == []

    def test_quest_with_priority(self):
        q = Quest(
            id="q1",
            task="Critical task",
            context="",
            category=Category.REASONING,
            priority=Priority.CRITICAL,
        )
        assert q.priority == Priority.CRITICAL

    def test_quest_with_dependencies(self):
        q = Quest(
            id="q3",
            task="Synthesize {q1} and {q2}",
            context="",
            category=Category.REASONING,
            depends_on=["q1", "q2"],
        )
        assert q.depends_on == ["q1", "q2"]


class TestValidation:
    def test_valid_dag(self):
        dag = QuestDAG(
            prompt="test",
            quests=[
                Quest(id="q1", task="A", context="", category=Category.SPEED),
                Quest(id="q2", task="B", context="", category=Category.CODE, depends_on=["q1"]),
            ],
        )
        validate_dag(dag)  # should not raise

    def test_duplicate_ids(self):
        dag = QuestDAG(
            prompt="test",
            quests=[
                Quest(id="q1", task="A", context="", category=Category.SPEED),
                Quest(id="q1", task="B", context="", category=Category.CODE),
            ],
        )
        with pytest.raises(QuestValidationError, match="Duplicate"):
            validate_dag(dag)

    def test_missing_dependency(self):
        dag = QuestDAG(
            prompt="test",
            quests=[
                Quest(id="q1", task="A", context="", category=Category.SPEED, depends_on=["q99"]),
            ],
        )
        with pytest.raises(QuestValidationError, match="unknown"):
            validate_dag(dag)

    def test_circular_dependency(self):
        dag = QuestDAG(
            prompt="test",
            quests=[
                Quest(id="q1", task="A", context="", category=Category.SPEED, depends_on=["q2"]),
                Quest(id="q2", task="B", context="", category=Category.CODE, depends_on=["q1"]),
            ],
        )
        with pytest.raises(QuestValidationError, match="Circular"):
            validate_dag(dag)


class TestComputeWaves:
    def test_single_wave(self):
        dag = QuestDAG(
            prompt="test",
            quests=[
                Quest(id="q1", task="A", context="", category=Category.SPEED),
                Quest(id="q2", task="B", context="", category=Category.CODE),
            ],
        )
        waves = compute_waves(dag)
        assert len(waves) == 1
        assert set(waves[0]) == {"q1", "q2"}

    def test_two_waves(self):
        dag = QuestDAG(
            prompt="test",
            quests=[
                Quest(id="q1", task="A", context="", category=Category.SPEED),
                Quest(id="q2", task="B", context="", category=Category.CODE),
                Quest(id="q3", task="C", context="", category=Category.REASONING, depends_on=["q1", "q2"]),
            ],
        )
        waves = compute_waves(dag)
        assert len(waves) == 2
        assert set(waves[0]) == {"q1", "q2"}
        assert waves[1] == ["q3"]

    def test_priority_ordering_within_wave(self):
        dag = QuestDAG(
            prompt="test",
            quests=[
                Quest(id="q1", task="A", context="", category=Category.SPEED, priority=Priority.LOW),
                Quest(id="q2", task="B", context="", category=Category.CODE, priority=Priority.CRITICAL),
                Quest(id="q3", task="C", context="", category=Category.REASONING, priority=Priority.HIGH),
            ],
        )
        waves = compute_waves(dag)
        assert len(waves) == 1
        assert waves[0] == ["q2", "q3", "q1"]  # critical, high, low


class TestFormatQuestBoard:
    def test_format_output(self):
        dag = QuestDAG(
            prompt="test prompt",
            quests=[
                Quest(id="q1", task="Do A", context="ctx", category=Category.SPEED),
                Quest(id="q2", task="Do B based on {q1}", context="", category=Category.REASONING, depends_on=["q1"]),
            ],
        )
        output = format_quest_board(dag)
        assert "q1" in output
        assert "q2" in output
        assert "speed" in output
        assert "reasoning" in output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_dag.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create `src/tavern/dag.py`**

```python
from enum import Enum
from pydantic import BaseModel


class Category(str, Enum):
    SPEED = "speed"
    CODE = "code"
    RESEARCH = "research"
    LARGE_CONTEXT = "large_context"
    REASONING = "reasoning"
    CREATIVE = "creative"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


PRIORITY_ORDER = {
    Priority.CRITICAL: 0,
    Priority.HIGH: 1,
    Priority.NORMAL: 2,
    Priority.LOW: 3,
}


class QuestValidationError(Exception):
    pass


class Quest(BaseModel):
    id: str
    task: str
    context: str
    category: Category
    priority: Priority = Priority.NORMAL
    depends_on: list[str] = []


class QuestDAG(BaseModel):
    prompt: str
    quests: list[Quest]


def validate_dag(dag: QuestDAG) -> None:
    quest_ids = [q.id for q in dag.quests]

    # Check duplicate IDs
    if len(quest_ids) != len(set(quest_ids)):
        seen = set()
        for qid in quest_ids:
            if qid in seen:
                raise QuestValidationError(f"Duplicate quest id: {qid}")
            seen.add(qid)

    id_set = set(quest_ids)

    # Check all dependencies reference existing quests
    for quest in dag.quests:
        for dep in quest.depends_on:
            if dep not in id_set:
                raise QuestValidationError(
                    f"Quest '{quest.id}' depends on unknown quest '{dep}'"
                )

    # Check for circular dependencies via topological sort
    in_degree = {q.id: len(q.depends_on) for q in dag.quests}
    deps_map = {q.id: set(q.depends_on) for q in dag.quests}
    queue = [qid for qid, deg in in_degree.items() if deg == 0]
    visited = 0

    while queue:
        current = queue.pop(0)
        visited += 1
        for quest in dag.quests:
            if current in deps_map[quest.id]:
                deps_map[quest.id].discard(current)
                in_degree[quest.id] -= 1
                if in_degree[quest.id] == 0:
                    queue.append(quest.id)

    if visited != len(dag.quests):
        raise QuestValidationError("Circular dependency detected in DAG")


def compute_waves(dag: QuestDAG) -> list[list[str]]:
    validate_dag(dag)
    quest_map = {q.id: q for q in dag.quests}
    remaining = {q.id: set(q.depends_on) for q in dag.quests}
    waves = []

    while remaining:
        wave = [qid for qid, deps in remaining.items() if len(deps) == 0]
        if not wave:
            raise QuestValidationError("Circular dependency detected in DAG")
        # Sort by priority (critical first), then by id for determinism
        wave.sort(key=lambda qid: (PRIORITY_ORDER[quest_map[qid].priority], qid))
        waves.append(wave)
        for qid in wave:
            del remaining[qid]
        for deps in remaining.values():
            deps -= set(wave)

    return waves


def format_quest_board(dag: QuestDAG) -> str:
    waves = compute_waves(dag)
    quest_map = {q.id: q for q in dag.quests}
    lines = []
    lines.append(f"Quest Board — {len(dag.quests)} quests, {len(waves)} waves")
    lines.append("")
    for i, wave in enumerate(waves):
        lines.append(f"  Wave {i + 1} ({'parallel' if len(wave) > 1 else 'sequential'}):")
        for qid in wave:
            quest = quest_map[qid]
            task_short = quest.task[:55] + "..." if len(quest.task) > 55 else quest.task
            deps = ", ".join(quest.depends_on) if quest.depends_on else "none"
            lines.append(f"    [{qid}] {task_short}")
            lines.append(
                f"         category={quest.category.value}  "
                f"priority={quest.priority.value}  "
                f"depends_on={deps}"
            )
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_dag.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/tavern/dag.py tests/test_dag.py
git commit -m "feat: DAG schema — Quest with context/priority, validation, priority-sorted waves"
```

---

### Task 5: Courtyard Module

**Files:**
- Create: `src/tavern/courtyard.py`
- Create: `tests/test_courtyard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_courtyard.py
from tavern.courtyard import Courtyard


class TestCourtyard:
    def test_prompt_only(self):
        c = Courtyard(prompt="Hello world")
        assert c.prompt == "Hello world"
        assert c.stdin_content is None
        assert c.has_stdin is False

    def test_with_stdin(self):
        c = Courtyard(prompt="Review this", stdin_content="def foo(): pass")
        assert c.has_stdin is True
        assert c.stdin_content == "def foo(): pass"

    def test_full_context(self):
        c = Courtyard(prompt="Review this", stdin_content="code here")
        full = c.full_context()
        assert "Review this" in full
        assert "code here" in full

    def test_full_context_no_stdin(self):
        c = Courtyard(prompt="Just a question")
        full = c.full_context()
        assert full == "Just a question"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_courtyard.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create `src/tavern/courtyard.py`**

```python
from dataclasses import dataclass


@dataclass
class Courtyard:
    prompt: str
    stdin_content: str | None = None

    @property
    def has_stdin(self) -> bool:
        return self.stdin_content is not None

    def full_context(self) -> str:
        if self.stdin_content:
            return f"{self.prompt}\n\n---\n\n{self.stdin_content}"
        return self.prompt
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_courtyard.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/tavern/courtyard.py tests/test_courtyard.py
git commit -m "feat: courtyard module — shared context space (prompt + stdin)"
```

---

### Task 6: Fallback Routing

**Files:**
- Create: `src/tavern/fallback.py`
- Create: `tests/test_fallback.py`

Updated from llmx with 9 providers and per-provider per-category model selection.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fallback.py
from tavern.fallback import (
    CATEGORY_RANKINGS,
    get_provider_chain,
    get_model_for_provider,
)


class TestCategoryRankings:
    def test_all_categories_present(self):
        expected = {"speed", "code", "research", "large_context", "reasoning", "creative"}
        assert set(CATEGORY_RANKINGS.keys()) == expected

    def test_ollama_is_last_in_every_category(self):
        for category, chain in CATEGORY_RANKINGS.items():
            assert chain[-1] == "ollama", f"ollama not last in {category}"

    def test_speed_first_pick_is_groq(self):
        assert CATEGORY_RANKINGS["speed"][0] == "groq"

    def test_code_first_pick_is_mistral(self):
        assert CATEGORY_RANKINGS["code"][0] == "mistral"


class TestGetProviderChain:
    def test_filters_by_available(self):
        chain = get_provider_chain("speed", ["groq", "ollama"])
        assert chain == ["groq", "ollama"]

    def test_preserves_ranking_order(self):
        chain = get_provider_chain("speed", ["ollama", "groq", "cerebras"])
        assert chain == ["groq", "cerebras", "ollama"]

    def test_skip_providers(self):
        chain = get_provider_chain("speed", ["groq", "cerebras", "ollama"], skip_providers=["groq"])
        assert chain == ["cerebras", "ollama"]

    def test_unknown_category_returns_empty(self):
        chain = get_provider_chain("unknown", ["groq", "ollama"])
        assert chain == []


class TestGetModelForProvider:
    def test_groq_speed(self):
        model = get_model_for_provider("groq", "speed")
        assert model == "llama-3.3-70b-versatile"

    def test_mistral_code(self):
        model = get_model_for_provider("mistral", "code")
        assert model == "codestral-latest"

    def test_gemini_large_context(self):
        model = get_model_for_provider("gemini", "large_context")
        assert model == "gemini-2.5-pro-preview-05-06"

    def test_openrouter_reasoning(self):
        model = get_model_for_provider("openrouter", "reasoning")
        assert model == "deepseek/deepseek-r1:free"

    def test_ollama_returns_none(self):
        model = get_model_for_provider("ollama", "speed")
        assert model is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fallback.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create `src/tavern/fallback.py`**

```python
CATEGORY_RANKINGS: dict[str, list[str]] = {
    "speed":         ["groq", "cerebras", "sambanova", "gemini", "ollama"],
    "code":          ["mistral", "openrouter", "github_models", "cerebras", "ollama"],
    "research":      ["gemini", "openrouter", "github_models", "sambanova", "ollama"],
    "large_context": ["gemini", "zhipu", "mistral", "github_models", "ollama"],
    "reasoning":     ["openrouter", "sambanova", "cerebras", "github_models", "ollama"],
    "creative":      ["github_models", "mistral", "sambanova", "gemini", "ollama"],
}

# Per-provider, per-category model selection
# If a provider+category combo is not listed, the provider's default_model is used.
PROVIDER_CATEGORY_MODELS: dict[str, dict[str, str]] = {
    "groq": {
        "speed": "llama-3.3-70b-versatile",
        "code": "qwen-qwq-32b",
        "research": "llama-3.3-70b-versatile",
        "large_context": "llama-3.3-70b-versatile",
        "reasoning": "qwen-qwq-32b",
        "creative": "llama-3.3-70b-versatile",
    },
    "cerebras": {
        "speed": "llama3.1-8b",
        "code": "qwen3-235b",
        "research": "qwen3-235b",
        "large_context": "qwen3-235b",
        "reasoning": "qwen3-235b",
        "creative": "qwen3-235b",
    },
    "sambanova": {
        "speed": "Meta-Llama-3.3-70B-Instruct",
        "code": "DeepSeek-R1-Distill-Llama-70B",
        "research": "Meta-Llama-3.1-405B-Instruct",
        "large_context": "Meta-Llama-3.1-405B-Instruct",
        "reasoning": "Meta-Llama-3.1-405B-Instruct",
        "creative": "Meta-Llama-3.1-405B-Instruct",
    },
    "gemini": {
        "speed": "gemini-2.0-flash",
        "code": "gemini-2.0-flash",
        "research": "gemini-2.5-pro-preview-05-06",
        "large_context": "gemini-2.5-pro-preview-05-06",
        "reasoning": "gemini-2.5-pro-preview-05-06",
        "creative": "gemini-2.5-pro-preview-05-06",
    },
    "openrouter": {
        "speed": "meta-llama/llama-3.3-70b-instruct:free",
        "code": "deepseek/deepseek-r1:free",
        "research": "deepseek/deepseek-r1:free",
        "large_context": "deepseek/deepseek-r1:free",
        "reasoning": "deepseek/deepseek-r1:free",
        "creative": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "mistral": {
        "speed": "mistral-small-latest",
        "code": "codestral-latest",
        "research": "mistral-large-latest",
        "large_context": "mistral-large-latest",
        "reasoning": "mistral-large-latest",
        "creative": "mistral-large-latest",
    },
    "github_models": {
        "speed": "meta-llama-3.3-70b-instruct",
        "code": "gpt-4o",
        "research": "gpt-4o",
        "large_context": "gpt-4o",
        "reasoning": "gpt-4o",
        "creative": "gpt-4o",
    },
    "zhipu": {
        "speed": "glm-4-flash",
        "code": "glm-4-flash",
        "research": "glm-4-flash",
        "large_context": "glm-4-flash",
        "reasoning": "glm-4-flash",
        "creative": "glm-4-flash",
    },
}


def get_provider_chain(
    category: str,
    available_providers: list[str],
    skip_providers: list[str] | None = None,
) -> list[str]:
    ranking = CATEGORY_RANKINGS.get(category, [])
    skip = set(skip_providers or [])
    return [p for p in ranking if p in available_providers and p not in skip]


def get_model_for_provider(provider: str, category: str) -> str | None:
    provider_models = PROVIDER_CATEGORY_MODELS.get(provider)
    if provider_models is None:
        return None  # ollama — use provider default
    return provider_models.get(category)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_fallback.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/tavern/fallback.py tests/test_fallback.py
git commit -m "feat: fallback routing — 9-provider category rankings with per-provider model selection"
```

---

### Task 7: OpenAI-Compatible Providers (Groq, Cerebras, SambaNova, Mistral, GitHub Models)

**Files:**
- Create: `src/tavern/providers/groq.py`
- Create: `src/tavern/providers/cerebras.py`
- Create: `src/tavern/providers/sambanova.py`
- Create: `src/tavern/providers/mistral.py`
- Create: `src/tavern/providers/github_models.py`
- Modify: `tests/test_providers.py`

All five use the same OpenAI-compatible chat completions API pattern. They differ only in URL, default model, and name.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_providers.py`:

```python
import httpx
import respx

from tavern.providers.groq import GroqProvider
from tavern.providers.cerebras import CerebrasProvider
from tavern.providers.sambanova import SambaNovaProvider
from tavern.providers.mistral import MistralProvider
from tavern.providers.github_models import GitHubModelsProvider


def _mock_openai_success(url: str) -> respx.Route:
    return respx.post(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "test response"}}],
            },
        )
    )


def _mock_openai_error(url: str, status: int = 429) -> respx.Route:
    return respx.post(url).mock(
        return_value=httpx.Response(status, text="Rate limited")
    )


class TestGroqProvider:
    @respx.mock
    def test_complete_success(self):
        _mock_openai_success("https://api.groq.com/openai/v1/chat/completions")
        provider = GroqProvider(api_key="test-key")
        result = provider.complete("Hello")
        assert result.success is True
        assert result.output == "test response"
        assert result.provider == "groq"

    @respx.mock
    def test_complete_error(self):
        _mock_openai_error("https://api.groq.com/openai/v1/chat/completions")
        provider = GroqProvider(api_key="test-key")
        result = provider.complete("Hello")
        assert result.success is False
        assert result.error_code == 429


class TestCerebrasProvider:
    @respx.mock
    def test_complete_success(self):
        _mock_openai_success("https://api.cerebras.ai/v1/chat/completions")
        provider = CerebrasProvider(api_key="test-key")
        result = provider.complete("Hello")
        assert result.success is True
        assert result.output == "test response"
        assert result.provider == "cerebras"


class TestSambaNovaProvider:
    @respx.mock
    def test_complete_success(self):
        _mock_openai_success("https://api.sambanova.ai/v1/chat/completions")
        provider = SambaNovaProvider(api_key="test-key")
        result = provider.complete("Hello")
        assert result.success is True
        assert result.provider == "sambanova"


class TestMistralProvider:
    @respx.mock
    def test_complete_success(self):
        _mock_openai_success("https://api.mistral.ai/v1/chat/completions")
        provider = MistralProvider(api_key="test-key")
        result = provider.complete("Hello")
        assert result.success is True
        assert result.provider == "mistral"


class TestGitHubModelsProvider:
    @respx.mock
    def test_complete_success(self):
        _mock_openai_success("https://models.inference.ai.azure.com/chat/completions")
        provider = GitHubModelsProvider(api_key="test-key")
        result = provider.complete("Hello")
        assert result.success is True
        assert result.provider == "github_models"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_providers.py -v`
Expected: FAIL (import errors)

- [ ] **Step 3: Create `src/tavern/providers/groq.py`**

```python
import time

import httpx

from .base import BaseProvider, ProviderResult

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(BaseProvider):
    name = "groq"
    default_model = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        start = time.monotonic()
        try:
            resp = httpx.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                },
                timeout=30.0,
            )
            latency = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return ProviderResult(
                    output="", success=False, provider=self.name, model=model,
                    latency_ms=latency, error_code=resp.status_code, error_message=resp.text,
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return ProviderResult(
                output=content, success=True, provider=self.name, model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="", success=False, provider=self.name, model=model,
                latency_ms=latency, error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        result = self.complete("Hi", model=self.default_model)
        return result.success
```

- [ ] **Step 4: Create `src/tavern/providers/cerebras.py`**

```python
import time

import httpx

from .base import BaseProvider, ProviderResult

CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"


class CerebrasProvider(BaseProvider):
    name = "cerebras"
    default_model = "llama3.1-8b"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        start = time.monotonic()
        try:
            resp = httpx.post(
                CEREBRAS_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                },
                timeout=30.0,
            )
            latency = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return ProviderResult(
                    output="", success=False, provider=self.name, model=model,
                    latency_ms=latency, error_code=resp.status_code, error_message=resp.text,
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return ProviderResult(
                output=content, success=True, provider=self.name, model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="", success=False, provider=self.name, model=model,
                latency_ms=latency, error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        result = self.complete("Hi", model=self.default_model)
        return result.success
```

- [ ] **Step 5: Create `src/tavern/providers/sambanova.py`**

```python
import time

import httpx

from .base import BaseProvider, ProviderResult

SAMBANOVA_API_URL = "https://api.sambanova.ai/v1/chat/completions"


class SambaNovaProvider(BaseProvider):
    name = "sambanova"
    default_model = "Meta-Llama-3.3-70B-Instruct"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        start = time.monotonic()
        try:
            resp = httpx.post(
                SAMBANOVA_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                },
                timeout=30.0,
            )
            latency = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return ProviderResult(
                    output="", success=False, provider=self.name, model=model,
                    latency_ms=latency, error_code=resp.status_code, error_message=resp.text,
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return ProviderResult(
                output=content, success=True, provider=self.name, model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="", success=False, provider=self.name, model=model,
                latency_ms=latency, error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        result = self.complete("Hi", model=self.default_model)
        return result.success
```

- [ ] **Step 6: Create `src/tavern/providers/mistral.py`**

```python
import time

import httpx

from .base import BaseProvider, ProviderResult

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


class MistralProvider(BaseProvider):
    name = "mistral"
    default_model = "mistral-large-latest"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        start = time.monotonic()
        try:
            resp = httpx.post(
                MISTRAL_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                },
                timeout=30.0,
            )
            latency = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return ProviderResult(
                    output="", success=False, provider=self.name, model=model,
                    latency_ms=latency, error_code=resp.status_code, error_message=resp.text,
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return ProviderResult(
                output=content, success=True, provider=self.name, model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="", success=False, provider=self.name, model=model,
                latency_ms=latency, error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        result = self.complete("Hi", model=self.default_model)
        return result.success
```

- [ ] **Step 7: Create `src/tavern/providers/github_models.py`**

```python
import time

import httpx

from .base import BaseProvider, ProviderResult

GITHUB_MODELS_API_URL = "https://models.inference.ai.azure.com/chat/completions"


class GitHubModelsProvider(BaseProvider):
    name = "github_models"
    default_model = "gpt-4o"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        start = time.monotonic()
        try:
            resp = httpx.post(
                GITHUB_MODELS_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                },
                timeout=30.0,
            )
            latency = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return ProviderResult(
                    output="", success=False, provider=self.name, model=model,
                    latency_ms=latency, error_code=resp.status_code, error_message=resp.text,
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return ProviderResult(
                output=content, success=True, provider=self.name, model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="", success=False, provider=self.name, model=model,
                latency_ms=latency, error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        result = self.complete("Hi", model=self.default_model)
        return result.success
```

- [ ] **Step 8: Run tests**

Run: `.venv/bin/pytest tests/test_providers.py -v`
Expected: 8 passed (3 base + 5 provider tests)

- [ ] **Step 9: Commit**

```bash
git add src/tavern/providers/groq.py src/tavern/providers/cerebras.py src/tavern/providers/sambanova.py src/tavern/providers/mistral.py src/tavern/providers/github_models.py tests/test_providers.py
git commit -m "feat: OpenAI-compatible providers — Groq, Cerebras, SambaNova, Mistral, GitHub Models"
```

---

### Task 8: Gemini, OpenRouter, Zhipu, and Ollama Providers

**Files:**
- Create: `src/tavern/providers/gemini.py`
- Create: `src/tavern/providers/openrouter.py`
- Create: `src/tavern/providers/zhipu.py`
- Create: `src/tavern/providers/ollama.py`
- Modify: `tests/test_providers.py`

These four have non-standard APIs (Gemini uses Google format, Zhipu uses a different endpoint, Ollama is local, OpenRouter uses GET for validation).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_providers.py`:

```python
from tavern.providers.gemini import GeminiProvider
from tavern.providers.openrouter import OpenRouterProvider
from tavern.providers.zhipu import ZhipuProvider
from tavern.providers.ollama import OllamaProvider


class TestGeminiProvider:
    @respx.mock
    def test_complete_success(self):
        respx.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent").mock(
            return_value=httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": "gemini response"}]}}],
                },
            )
        )
        provider = GeminiProvider(api_key="test-key")
        result = provider.complete("Hello")
        assert result.success is True
        assert result.output == "gemini response"
        assert result.provider == "gemini"


class TestOpenRouterProvider:
    @respx.mock
    def test_complete_success(self):
        _mock_openai_success("https://openrouter.ai/api/v1/chat/completions")
        provider = OpenRouterProvider(api_key="test-key")
        result = provider.complete("Hello")
        assert result.success is True
        assert result.provider == "openrouter"

    @respx.mock
    def test_validate_success(self):
        respx.get("https://openrouter.ai/api/v1/auth/key").mock(
            return_value=httpx.Response(200, json={})
        )
        provider = OpenRouterProvider(api_key="test-key")
        assert provider.validate() is True


class TestZhipuProvider:
    @respx.mock
    def test_complete_success(self):
        respx.post("https://open.bigmodel.cn/api/paas/v4/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "zhipu response"}}],
                },
            )
        )
        provider = ZhipuProvider(api_key="test-key")
        result = provider.complete("Hello")
        assert result.success is True
        assert result.output == "zhipu response"
        assert result.provider == "zhipu"


class TestOllamaProvider:
    @respx.mock
    def test_complete_success(self):
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "local response"})
        )
        provider = OllamaProvider()
        result = provider.complete("Hello")
        assert result.success is True
        assert result.output == "local response"
        assert result.provider == "ollama"

    @respx.mock
    def test_validate_with_models(self):
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "llama3"}]})
        )
        provider = OllamaProvider()
        is_valid, models = provider.validate_with_models()
        assert is_valid is True
        assert "llama3" in models
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_providers.py -v`
Expected: FAIL (import errors)

- [ ] **Step 3: Create `src/tavern/providers/gemini.py`**

```python
import time

import httpx

from .base import BaseProvider, ProviderResult

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(BaseProvider):
    name = "gemini"
    default_model = "gemini-2.0-flash"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        url = f"{GEMINI_API_BASE}/{model}:generateContent"
        start = time.monotonic()
        try:
            resp = httpx.post(
                url,
                params={"key": self.api_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30.0,
            )
            latency = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return ProviderResult(
                    output="", success=False, provider=self.name, model=model,
                    latency_ms=latency, error_code=resp.status_code, error_message=resp.text,
                )

            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            return ProviderResult(
                output=content, success=True, provider=self.name, model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="", success=False, provider=self.name, model=model,
                latency_ms=latency, error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        result = self.complete("Hi", model=self.default_model)
        return result.success
```

- [ ] **Step 4: Create `src/tavern/providers/openrouter.py`**

```python
import time

import httpx

from .base import BaseProvider, ProviderResult

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_AUTH_URL = "https://openrouter.ai/api/v1/auth/key"


class OpenRouterProvider(BaseProvider):
    name = "openrouter"
    default_model = "deepseek/deepseek-r1:free"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        start = time.monotonic()
        try:
            resp = httpx.post(
                OPENROUTER_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                },
                timeout=30.0,
            )
            latency = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return ProviderResult(
                    output="", success=False, provider=self.name, model=model,
                    latency_ms=latency, error_code=resp.status_code, error_message=resp.text,
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return ProviderResult(
                output=content, success=True, provider=self.name, model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="", success=False, provider=self.name, model=model,
                latency_ms=latency, error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        try:
            resp = httpx.get(
                OPENROUTER_AUTH_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
```

- [ ] **Step 5: Create `src/tavern/providers/zhipu.py`**

```python
import time

import httpx

from .base import BaseProvider, ProviderResult

ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


class ZhipuProvider(BaseProvider):
    name = "zhipu"
    default_model = "glm-4-flash"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        start = time.monotonic()
        try:
            resp = httpx.post(
                ZHIPU_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                },
                timeout=30.0,
            )
            latency = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return ProviderResult(
                    output="", success=False, provider=self.name, model=model,
                    latency_ms=latency, error_code=resp.status_code, error_message=resp.text,
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return ProviderResult(
                output=content, success=True, provider=self.name, model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="", success=False, provider=self.name, model=model,
                latency_ms=latency, error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        result = self.complete("Hi", model=self.default_model)
        return result.success
```

- [ ] **Step 6: Create `src/tavern/providers/ollama.py`**

```python
import time

import httpx

from .base import BaseProvider, ProviderResult

OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    name = "ollama"
    default_model = "llama3"

    def __init__(self, model: str | None = None):
        self.default_model = model or "llama3"

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        start = time.monotonic()
        try:
            resp = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60.0,
            )
            latency = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return ProviderResult(
                    output="", success=False, provider=self.name, model=model,
                    latency_ms=latency, error_code=resp.status_code, error_message=resp.text,
                )

            data = resp.json()
            return ProviderResult(
                output=data["response"], success=True, provider=self.name, model=model,
                latency_ms=latency,
            )
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="", success=False, provider=self.name, model=model,
                latency_ms=latency, error_message=str(e),
            )

    def validate(self) -> bool:
        is_valid, _ = self.validate_with_models()
        return is_valid

    def validate_with_models(self) -> tuple[bool, list[str]]:
        try:
            resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if resp.status_code != 200:
                return False, []
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return True, models
        except (httpx.ConnectError, httpx.TimeoutException):
            return False, []
```

- [ ] **Step 7: Run tests**

Run: `.venv/bin/pytest tests/test_providers.py -v`
Expected: 14 passed

- [ ] **Step 8: Commit**

```bash
git add src/tavern/providers/gemini.py src/tavern/providers/openrouter.py src/tavern/providers/zhipu.py src/tavern/providers/ollama.py tests/test_providers.py
git commit -m "feat: remaining providers — Gemini, OpenRouter, Zhipu, Ollama"
```

---

### Task 9: Provider Registry

**Files:**
- Modify: `src/tavern/providers/__init__.py`
- Modify: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_providers.py`:

```python
from tavern.providers import PROVIDER_REGISTRY


class TestProviderRegistry:
    def test_all_nine_providers_registered(self):
        expected = {
            "groq", "cerebras", "sambanova", "gemini",
            "openrouter", "mistral", "github_models", "zhipu", "ollama",
        }
        assert set(PROVIDER_REGISTRY.keys()) == expected

    def test_all_are_base_provider_subclasses(self):
        for name, cls in PROVIDER_REGISTRY.items():
            assert issubclass(cls, BaseProvider), f"{name} is not a BaseProvider"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_providers.py::TestProviderRegistry -v`
Expected: FAIL (PROVIDER_REGISTRY not defined or incomplete)

- [ ] **Step 3: Update `src/tavern/providers/__init__.py`**

```python
from .groq import GroqProvider
from .cerebras import CerebrasProvider
from .sambanova import SambaNovaProvider
from .gemini import GeminiProvider
from .openrouter import OpenRouterProvider
from .mistral import MistralProvider
from .github_models import GitHubModelsProvider
from .zhipu import ZhipuProvider
from .ollama import OllamaProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqProvider,
    "cerebras": CerebrasProvider,
    "sambanova": SambaNovaProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
    "mistral": MistralProvider,
    "github_models": GitHubModelsProvider,
    "zhipu": ZhipuProvider,
    "ollama": OllamaProvider,
}
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_providers.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add src/tavern/providers/__init__.py tests/test_providers.py
git commit -m "feat: provider registry — all 9 adventurers registered"
```

---

### Task 10: Guild Master — Decompose

**Files:**
- Create: `src/tavern/guildmaster.py`
- Create: `tests/test_guildmaster.py`

Uses the Anthropic Python SDK to call Claude for decomposition (Courtyard → QuestDAG).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_guildmaster.py
import json
from unittest.mock import MagicMock, patch

from tavern.courtyard import Courtyard
from tavern.dag import Category, Priority, QuestDAG
from tavern.guildmaster import GuildMaster, build_decompose_prompt, parse_dag_response


SAMPLE_DAG_JSON = json.dumps({
    "prompt": "Compare REST vs GraphQL",
    "quests": [
        {
            "id": "q1",
            "task": "List REST pros/cons for mobile",
            "context": "Focus on mobile app backends",
            "category": "speed",
            "priority": "high",
            "depends_on": [],
        },
        {
            "id": "q2",
            "task": "List GraphQL pros/cons for mobile",
            "context": "Focus on mobile app backends",
            "category": "speed",
            "priority": "high",
            "depends_on": [],
        },
        {
            "id": "q3",
            "task": "Create comparison table from {q1} and {q2}",
            "context": "",
            "category": "reasoning",
            "priority": "critical",
            "depends_on": ["q1", "q2"],
        },
    ],
})


class TestBuildDecomposePrompt:
    def test_includes_courtyard_content(self):
        courtyard = Courtyard(prompt="Compare REST vs GraphQL")
        prompt = build_decompose_prompt(courtyard)
        assert "Compare REST vs GraphQL" in prompt
        assert "category" in prompt.lower()
        assert "priority" in prompt.lower()
        assert "context" in prompt.lower()

    def test_includes_stdin_when_present(self):
        courtyard = Courtyard(prompt="Review this", stdin_content="def foo(): pass")
        prompt = build_decompose_prompt(courtyard)
        assert "def foo(): pass" in prompt


class TestParseDagResponse:
    def test_parse_valid_json(self):
        dag = parse_dag_response(SAMPLE_DAG_JSON)
        assert isinstance(dag, QuestDAG)
        assert len(dag.quests) == 3
        assert dag.quests[0].category == Category.SPEED
        assert dag.quests[2].priority == Priority.CRITICAL
        assert dag.quests[2].depends_on == ["q1", "q2"]

    def test_parse_json_in_code_block(self):
        response = f"Here is the DAG:\n```json\n{SAMPLE_DAG_JSON}\n```"
        dag = parse_dag_response(response)
        assert len(dag.quests) == 3

    def test_parse_invalid_json_raises(self):
        import pytest
        with pytest.raises(ValueError):
            parse_dag_response("not json at all")


class TestGuildMasterDecompose:
    @patch("tavern.guildmaster.anthropic")
    def test_decompose_calls_api(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=SAMPLE_DAG_JSON)]
        )

        gm = GuildMaster(api_key="sk-ant-test")
        courtyard = Courtyard(prompt="Compare REST vs GraphQL")
        dag = gm.decompose(courtyard)

        assert isinstance(dag, QuestDAG)
        assert len(dag.quests) == 3
        mock_client.messages.create.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_guildmaster.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create `src/tavern/guildmaster.py`**

```python
import json
import re

import anthropic

from .courtyard import Courtyard
from .dag import QuestDAG, validate_dag

DECOMPOSE_SYSTEM_PROMPT = """\
You are the Guild Master of the Tavern — an orchestrator that decomposes complex prompts into a DAG of quests for parallel execution by different LLM providers (Adventurers).

Your job: analyze the user's request (the Courtyard) and produce a JSON quest DAG.

## Quest Schema

Each quest has:
- id: unique identifier (q1, q2, q3...)
- task: clear instruction for the adventurer
- context: the CURATED slice of input relevant to this quest. Include ONLY what the adventurer needs. This is the key — don't copy the full input to every quest.
- category: one of: speed, code, research, large_context, reasoning, creative
- priority: one of: critical, high, normal, low
- depends_on: list of quest IDs this quest needs. Use {qX} in the task to reference another quest's output.

## Categories
- speed: quick lookups, translations, short summaries
- code: code generation, debugging, refactoring
- research: questions needing current/web-grounded information
- large_context: tasks involving long documents
- reasoning: analysis, comparison, logic, math
- creative: writing, brainstorming, ideation

## Rules
1. Maximize parallelism — independent quests should have no dependencies
2. Curate context — each quest gets only the input slice it needs
3. Use {qX} placeholders to pass outputs between quests
4. Final synthesis quests should depend on earlier quests and have category "reasoning"
5. Keep quests focused — one clear objective per quest

## Output Format
Respond with ONLY a JSON object (no markdown, no explanation):
{
  "prompt": "original user prompt",
  "quests": [...]
}
"""


def build_decompose_prompt(courtyard: Courtyard) -> str:
    parts = [f"User prompt: {courtyard.prompt}"]
    if courtyard.has_stdin:
        parts.append(f"\nPiped input:\n{courtyard.stdin_content}")
    parts.append(
        "\nDecompose this into a quest DAG. "
        "Curate the context field for each quest — include only what that specific adventurer needs. "
        "Assign appropriate category and priority to each quest."
    )
    return "\n".join(parts)


def parse_dag_response(response_text: str) -> QuestDAG:
    text = response_text.strip()

    # Try to extract JSON from code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Guild Master response as JSON: {e}")

    dag = QuestDAG(**data)
    validate_dag(dag)
    return dag


class GuildMaster:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250514",
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def decompose(self, courtyard: Courtyard) -> QuestDAG:
        user_prompt = build_decompose_prompt(courtyard)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=DECOMPOSE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return parse_dag_response(response.content[0].text)

    def synthesize(self, courtyard: Courtyard, quest_results: dict[str, str]) -> str:
        results_text = "\n\n".join(
            f"## Quest {qid}\n{output}" for qid, output in quest_results.items()
        )
        user_prompt = (
            f"Original request: {courtyard.prompt}\n\n"
            f"Quest results:\n{results_text}\n\n"
            "Synthesize these quest results into a clear, unified answer. "
            "If any quest result is low quality or missing, handle it gracefully "
            "using your own knowledge."
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_guildmaster.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/tavern/guildmaster.py tests/test_guildmaster.py
git commit -m "feat: guild master — decompose (Courtyard → DAG) and synthesize via Anthropic API"
```

---

### Task 11: Guild Master — Synthesize

**Files:**
- Modify: `tests/test_guildmaster.py`

The `synthesize()` method was already created in Task 10. This task adds its test.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_guildmaster.py`:

```python
class TestGuildMasterSynthesize:
    @patch("tavern.guildmaster.anthropic")
    def test_synthesize_calls_api(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Here is the unified answer...")]
        )

        gm = GuildMaster(api_key="sk-ant-test")
        courtyard = Courtyard(prompt="Compare REST vs GraphQL")
        result = gm.synthesize(courtyard, {
            "q1": "REST is simple and cacheable...",
            "q2": "GraphQL is flexible and reduces over-fetching...",
        })

        assert "unified answer" in result
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        assert "REST" in str(call_kwargs)
        assert "GraphQL" in str(call_kwargs)
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/pytest tests/test_guildmaster.py -v`
Expected: 6 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_guildmaster.py
git commit -m "test: guild master synthesize — verify API call with quest results"
```

---

### Task 12: DAG Executor

**Files:**
- Create: `src/tavern/executor.py`
- Create: `tests/test_executor.py`

Priority-aware wave executor with ThreadPoolExecutor. Adapted from `~/llmx/src/llmx/executor.py` — adds priority ordering and context-aware prompts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_executor.py
from unittest.mock import MagicMock

from tavern.dag import Category, Priority, Quest, QuestDAG
from tavern.executor import DagExecutor, QuestResult, ExecutionResult
from tavern.providers.base import ProviderResult


def _make_provider(name: str, output: str = "response", success: bool = True) -> MagicMock:
    provider = MagicMock()
    provider.name = name
    provider.complete.return_value = ProviderResult(
        output=output, success=success, provider=name, model="test", latency_ms=50,
    )
    return provider


class TestQuestResult:
    def test_success_result(self):
        r = QuestResult(output="ok", success=True, provider="groq", model="llama", latency_ms=100)
        assert r.success is True
        assert r.fallback_exhausted is False


class TestDagExecutor:
    def test_single_quest(self):
        dag = QuestDAG(
            prompt="test",
            quests=[Quest(id="q1", task="Do A", context="ctx", category=Category.SPEED)],
        )
        providers = {"groq": _make_provider("groq")}
        executor = DagExecutor(dag, providers, ["groq"])
        result = executor.execute()
        assert "q1" in result.results
        assert result.results["q1"].success is True

    def test_parallel_quests(self):
        dag = QuestDAG(
            prompt="test",
            quests=[
                Quest(id="q1", task="A", context="", category=Category.SPEED),
                Quest(id="q2", task="B", context="", category=Category.SPEED),
            ],
        )
        providers = {"groq": _make_provider("groq")}
        executor = DagExecutor(dag, providers, ["groq"])
        result = executor.execute()
        assert len(result.results) == 2
        assert result.results["q1"].success
        assert result.results["q2"].success

    def test_dependent_quests_with_placeholder(self):
        dag = QuestDAG(
            prompt="test",
            quests=[
                Quest(id="q1", task="A", context="", category=Category.SPEED),
                Quest(id="q2", task="Combine {q1}", context="", category=Category.REASONING, depends_on=["q1"]),
            ],
        )
        groq = _make_provider("groq", output="result-A")
        openrouter = _make_provider("openrouter", output="combined")
        providers = {"groq": groq, "openrouter": openrouter}
        executor = DagExecutor(dag, providers, ["groq", "openrouter"])
        result = executor.execute()
        assert result.results["q2"].success

    def test_context_included_in_prompt(self):
        dag = QuestDAG(
            prompt="test",
            quests=[Quest(id="q1", task="Analyze code", context="def foo(): pass", category=Category.CODE)],
        )
        mistral = _make_provider("mistral")
        providers = {"mistral": mistral}
        executor = DagExecutor(dag, providers, ["mistral"])
        executor.execute()
        call_args = mistral.complete.call_args
        prompt_sent = call_args[0][0] if call_args[0] else call_args[1]["prompt"]
        assert "def foo(): pass" in prompt_sent

    def test_fallback_on_failure(self):
        dag = QuestDAG(
            prompt="test",
            quests=[Quest(id="q1", task="A", context="", category=Category.SPEED)],
        )
        groq_fail = _make_provider("groq", success=False)
        cerebras_ok = _make_provider("cerebras", output="fallback-ok")
        providers = {"groq": groq_fail, "cerebras": cerebras_ok}
        executor = DagExecutor(dag, providers, ["groq", "cerebras"])
        result = executor.execute()
        assert result.results["q1"].success is True
        assert result.results["q1"].provider == "cerebras"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_executor.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create `src/tavern/executor.py`**

```python
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Event

from .dag import QuestDAG, Quest, compute_waves
from .fallback import get_provider_chain, get_model_for_provider
from .providers.base import BaseProvider


@dataclass
class QuestResult:
    output: str
    success: bool
    provider: str
    model: str
    latency_ms: int = 0
    error_message: str = ""
    fallback_exhausted: bool = False


@dataclass
class ExecutionResult:
    results: dict[str, QuestResult] = field(default_factory=dict)
    execution_time_ms: int = 0
    retries: dict[str, list[str]] = field(default_factory=dict)


class DagExecutor:
    def __init__(
        self,
        dag: QuestDAG,
        providers: dict[str, BaseProvider],
        available_providers: list[str],
        max_workers: int = 8,
        on_quest_complete: callable = None,
    ):
        self.dag = dag
        self.providers = providers
        self.available_providers = available_providers
        self.max_workers = max_workers
        self.on_quest_complete = on_quest_complete
        self.quest_outputs: dict[str, str] = {}
        self.quest_events: dict[str, Event] = {q.id: Event() for q in dag.quests}

    def _substitute_placeholders(self, task: str) -> str:
        def replacer(match: re.Match) -> str:
            quest_id = match.group(1)
            return self.quest_outputs.get(quest_id, f"{{{quest_id}}}")
        return re.sub(r"\{(q\d+)\}", replacer, task)

    def _build_prompt(self, quest: Quest) -> str:
        task = self._substitute_placeholders(quest.task)
        if quest.context:
            return f"{task}\n\nContext:\n{quest.context}"
        return task

    def _execute_quest(self, quest: Quest) -> QuestResult:
        # Wait for dependencies
        for dep_id in quest.depends_on:
            self.quest_events[dep_id].wait()

        prompt = self._build_prompt(quest)
        chain = get_provider_chain(quest.category.value, self.available_providers)
        retries = []

        for provider_name in chain:
            provider = self.providers[provider_name]
            model = get_model_for_provider(provider_name, quest.category.value)
            result = provider.complete(prompt, model=model)

            if result.success:
                self.quest_outputs[quest.id] = result.output
                self.quest_events[quest.id].set()
                if retries:
                    self.retries[quest.id] = retries
                qr = QuestResult(
                    output=result.output,
                    success=True,
                    provider=result.provider,
                    model=result.model,
                    latency_ms=result.latency_ms,
                )
                if self.on_quest_complete:
                    self.on_quest_complete(quest.id, qr)
                return qr
            else:
                retries.append(f"{provider_name}:{result.error_code or 'error'}")

        # All providers failed
        self.quest_outputs[quest.id] = ""
        self.quest_events[quest.id].set()
        if retries:
            self.retries[quest.id] = retries
        qr = QuestResult(
            output="",
            success=False,
            provider="none",
            model="none",
            error_message="All providers failed",
            fallback_exhausted=True,
        )
        if self.on_quest_complete:
            self.on_quest_complete(quest.id, qr)
        return qr

    def execute(self) -> ExecutionResult:
        self.retries: dict[str, list[str]] = {}
        start = time.monotonic()
        results: dict[str, QuestResult] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for quest in self.dag.quests:
                future = pool.submit(self._execute_quest, quest)
                futures[future] = quest.id

            for future in as_completed(futures):
                quest_id = futures[future]
                results[quest_id] = future.result()

        execution_time = int((time.monotonic() - start) * 1000)
        return ExecutionResult(
            results=results,
            execution_time_ms=execution_time,
            retries=self.retries,
        )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_executor.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/tavern/executor.py tests/test_executor.py
git commit -m "feat: DAG executor — priority-aware parallel execution with fallback chains"
```

---

### Task 13: Setup Wizard

**Files:**
- Create: `src/tavern/wizard.py`
- Create: `tests/test_wizard.py`

Interactive rich wizard: Anthropic key first, then 8 adventurer API keys, then Ollama auto-detect.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wizard.py
from unittest.mock import patch, MagicMock

from tavern.wizard import ADVENTURER_INFO, run_wizard


class TestWizardInfo:
    def test_adventurer_info_has_eight_entries(self):
        assert len(ADVENTURER_INFO) == 8  # all except Ollama

    def test_all_have_required_fields(self):
        for info in ADVENTURER_INFO:
            assert "name" in info
            assert "label" in info
            assert "description" in info
            assert "url" in info


class TestRunWizard:
    @patch("tavern.wizard.OllamaProvider")
    @patch("tavern.wizard.Prompt")
    @patch("tavern.wizard.console")
    @patch("tavern.wizard.save_config")
    def test_wizard_skips_all(self, mock_save, mock_console, mock_prompt, mock_ollama):
        # Simulate pressing Enter (skip) for every prompt
        mock_prompt.ask.return_value = ""
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.validate_with_models.return_value = (False, [])
        mock_ollama.return_value = mock_ollama_instance

        result = run_wizard()
        assert "active_adventurers" in result
        assert result["active_adventurers"] == []
        mock_save.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_wizard.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create `src/tavern/wizard.py`**

```python
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

from .config import TavernConfig, save_config, DEFAULT_CONFIG_PATH
from .providers.groq import GroqProvider
from .providers.cerebras import CerebrasProvider
from .providers.sambanova import SambaNovaProvider
from .providers.gemini import GeminiProvider
from .providers.openrouter import OpenRouterProvider
from .providers.mistral import MistralProvider
from .providers.github_models import GitHubModelsProvider
from .providers.zhipu import ZhipuProvider
from .providers.ollama import OllamaProvider

console = Console()

ADVENTURER_INFO = [
    {
        "name": "groq",
        "label": "Groq",
        "description": "Fastest inference — Llama, DeepSeek, Qwen",
        "url": "https://console.groq.com/keys",
        "cls": GroqProvider,
    },
    {
        "name": "cerebras",
        "label": "Cerebras",
        "description": "Ultra-fast on wafer-scale — Llama, Qwen",
        "url": "https://cloud.cerebras.ai/",
        "cls": CerebrasProvider,
    },
    {
        "name": "sambanova",
        "label": "SambaNova",
        "description": "Access to 405B params — Llama 3.1 405B",
        "url": "https://cloud.sambanova.ai/",
        "cls": SambaNovaProvider,
    },
    {
        "name": "gemini",
        "label": "Google Gemini",
        "description": "1M context, multimodal, search grounding",
        "url": "https://aistudio.google.com/apikey",
        "cls": GeminiProvider,
    },
    {
        "name": "openrouter",
        "label": "OpenRouter",
        "description": "Meta-provider — DeepSeek R1, Llama, Gemma free",
        "url": "https://openrouter.ai/keys",
        "cls": OpenRouterProvider,
    },
    {
        "name": "mistral",
        "label": "Mistral",
        "description": "Best free code model (Codestral) + Mistral Large",
        "url": "https://console.mistral.ai/api-keys",
        "cls": MistralProvider,
    },
    {
        "name": "github_models",
        "label": "GitHub Models",
        "description": "Free GPT-4o, DeepSeek-R1, Llama",
        "url": "https://github.com/settings/tokens",
        "cls": GitHubModelsProvider,
    },
    {
        "name": "zhipu",
        "label": "Zhipu (GLM)",
        "description": "GLM-4-Flash free, 203K context",
        "url": "https://open.bigmodel.cn/",
        "cls": ZhipuProvider,
    },
]


def run_wizard() -> dict:
    console.print(Panel(
        Text("Tavern Setup Wizard", style="bold cyan", justify="center"),
        subtitle="Configure your Guild Master and Adventurers",
    ))
    console.print()

    config = TavernConfig()
    total = len(ADVENTURER_INFO) + 2  # +1 for Anthropic, +1 for Ollama
    active_count = 0

    # Step 1: Anthropic API key (Guild Master)
    console.print(f"[bold][1/{total}] Anthropic API Key[/bold] — Guild Master (Claude)")
    console.print("       Get a key at: [link]https://console.anthropic.com/settings/keys[/link]")
    console.print("       (Or set ANTHROPIC_API_KEY env var)")

    anthropic_key = Prompt.ask(
        "       Enter key (or press Enter to skip)",
        default="",
        show_default=False,
    )

    if anthropic_key:
        console.print("       Validating...", end=" ")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            client.messages.create(
                model="claude-sonnet-4-5-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            console.print("[green]Validated[/green]")
            config.guild_master = {"api_key": anthropic_key, "model": "claude-sonnet-4-5-20250514"}
        except Exception:
            console.print("[red]Failed[/red] — check your key")
            config.guild_master = {}
    else:
        console.print("       [dim]Skipped — will use ANTHROPIC_API_KEY env var[/dim]")

    console.print()

    # Step 2: Adventurer API keys
    for i, info in enumerate(ADVENTURER_INFO, 2):
        console.print(f"[bold][{i}/{total}] {info['label']}[/bold] — {info['description']}")
        console.print(f"       Get a key at: [link]{info['url']}[/link]")

        key = Prompt.ask(
            "       Enter key (or press Enter to skip)",
            default="",
            show_default=False,
        )

        if not key:
            console.print("       [dim]Skipped[/dim]")
            config.adventurers[info["name"]] = {"api_key": None, "validated": False}
        else:
            console.print("       Validating...", end=" ")
            provider = info["cls"](api_key=key)
            valid = provider.validate()

            if valid:
                console.print("[green]Validated[/green]")
                config.adventurers[info["name"]] = {"api_key": key, "validated": True}
                active_count += 1
            else:
                console.print("[red]Failed[/red] — key will not be used")
                config.adventurers[info["name"]] = {"api_key": key, "validated": False}

        console.print()

    # Step 3: Ollama
    console.print(f"[bold][{total}/{total}] Ollama[/bold] — Local fallback, no key needed")
    console.print("       Checking localhost:11434...", end=" ")
    ollama = OllamaProvider()
    is_valid, models = ollama.validate_with_models()

    if is_valid:
        console.print(f"[green]Running[/green] — models: {', '.join(models)}")
        config.adventurers["ollama"] = {"available": True, "models": models}
        active_count += 1
    else:
        console.print("[yellow]Not running[/yellow]")
        config.adventurers["ollama"] = {"available": False, "models": []}

    console.print()
    save_config(config)
    console.print(Panel(
        f"[bold green]{active_count}/{total - 1} adventurers active[/bold green]\n"
        f"Config saved to {DEFAULT_CONFIG_PATH}",
        title="Setup Complete",
    ))

    return {
        "active_adventurers": config.get_available_adventurers(),
        "config_path": str(DEFAULT_CONFIG_PATH),
    }
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_wizard.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/tavern/wizard.py tests/test_wizard.py
git commit -m "feat: setup wizard — Anthropic key + 8 adventurers + Ollama auto-detect"
```

---

### Task 14: CLI Module

**Files:**
- Create: `src/tavern/cli.py`
- Create: `tests/test_cli.py`

Rich live display, arg parsing, main flow (read stdin → build Courtyard → decompose → execute → synthesize → print).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import sys
from unittest.mock import patch, MagicMock

from tavern.cli import parse_args, build_courtyard


class TestParseArgs:
    def test_basic_prompt(self):
        args = parse_args(["Compare REST vs GraphQL"])
        assert args.prompt == "Compare REST vs GraphQL"
        assert args.model is None

    def test_with_model(self):
        args = parse_args(["--model", "claude-opus-4-0-20250514", "Hello"])
        assert args.model == "claude-opus-4-0-20250514"
        assert args.prompt == "Hello"

    def test_setup_command(self):
        args = parse_args(["setup"])
        assert args.command == "setup"


class TestBuildCourtyard:
    def test_prompt_only(self):
        courtyard = build_courtyard("Hello world", stdin_text=None)
        assert courtyard.prompt == "Hello world"
        assert courtyard.stdin_content is None

    def test_with_stdin(self):
        courtyard = build_courtyard("Review", stdin_text="code here")
        assert courtyard.stdin_content == "code here"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create `src/tavern/cli.py`**

```python
import argparse
import sys
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import load_config
from .courtyard import Courtyard
from .dag import QuestDAG, Quest, compute_waves
from .executor import DagExecutor, QuestResult
from .fallback import get_provider_chain
from .guildmaster import GuildMaster
from .providers import PROVIDER_REGISTRY
from .providers.base import BaseProvider

console = Console()

STATUS_ICONS = {
    "waiting": "[dim]◇[/dim]",
    "running": "[yellow]⟳[/yellow]",
    "done": "[green]✓[/green]",
    "failed": "[red]✗[/red]",
    "retrying": "[yellow]↻[/yellow]",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tavern",
        description="Multi-LLM orchestrator — Guild Master decomposes, Adventurers execute",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("setup", help="Run the setup wizard")

    parser.add_argument("prompt", nargs="?", help="The prompt to decompose and execute")
    parser.add_argument("--model", help="Override Guild Master model (default: claude-sonnet-4-5-20250514)")

    return parser.parse_args(argv)


def build_courtyard(prompt: str, stdin_text: str | None) -> Courtyard:
    return Courtyard(prompt=prompt, stdin_content=stdin_text)


def build_providers(config) -> dict[str, BaseProvider]:
    providers = {}
    for name, info in config.adventurers.items():
        if name == "ollama":
            if info.get("available", False):
                cls = PROVIDER_REGISTRY["ollama"]
                providers["ollama"] = cls()
        elif info.get("validated", False) and info.get("api_key"):
            cls = PROVIDER_REGISTRY.get(name)
            if cls:
                providers[name] = cls(api_key=info["api_key"])
    return providers


def build_quest_table(dag: QuestDAG, statuses: dict[str, str]) -> Table:
    table = Table(title="Quest Board", show_header=True, header_style="bold cyan")
    table.add_column("Quest", style="bold", width=6)
    table.add_column("Task", width=40)
    table.add_column("Cat.", width=10)
    table.add_column("Pri.", width=6)
    table.add_column("", width=3)

    for quest in dag.quests:
        status = statuses.get(quest.id, "waiting")
        icon = STATUS_ICONS.get(status, "?")
        task_short = quest.task[:38] + ".." if len(quest.task) > 38 else quest.task
        table.add_row(
            quest.id,
            task_short,
            quest.category.value,
            quest.priority.value[:4],
            icon,
        )
    return table


def run_quest(prompt: str, model: str | None = None) -> None:
    config = load_config()

    # Check Anthropic key
    anthropic_key = config.get_anthropic_key()
    if not anthropic_key:
        console.print("[red]No Anthropic API key found.[/red]")
        console.print("Run [bold]tavern setup[/bold] or set ANTHROPIC_API_KEY env var.")
        sys.exit(1)

    # Check adventurers
    providers = build_providers(config)
    available = list(providers.keys())
    if not available:
        console.print("[red]No adventurers configured.[/red]")
        console.print("Run [bold]tavern setup[/bold] to configure API keys.")
        sys.exit(1)

    gm_model = model or config.get_guild_master_model()

    # Read stdin if piped
    stdin_text = None
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read()

    courtyard = build_courtyard(prompt, stdin_text)

    # Decompose
    console.print(Panel("[bold cyan]Guild Master is analyzing your quest...[/bold cyan]"))
    gm = GuildMaster(api_key=anthropic_key, model=gm_model)
    dag = gm.decompose(courtyard)

    # Set up live display
    statuses = {q.id: "waiting" for q in dag.quests}
    quest_results: dict[str, QuestResult] = {}

    def on_quest_complete(quest_id: str, result: QuestResult):
        statuses[quest_id] = "done" if result.success else "failed"
        quest_results[quest_id] = result

    # Show DAG and execute
    with Live(build_quest_table(dag, statuses), console=console, refresh_per_second=4) as live:
        # Mark running
        for quest in dag.quests:
            if not quest.depends_on:
                statuses[quest.id] = "running"
        live.update(build_quest_table(dag, statuses))

        executor = DagExecutor(
            dag, providers, available,
            on_quest_complete=on_quest_complete,
        )
        execution_result = executor.execute()

        # Final update
        for qid, qr in execution_result.results.items():
            statuses[qid] = "done" if qr.success else "failed"
        live.update(build_quest_table(dag, statuses))

    # Synthesize
    console.print()
    console.print(Panel("[bold cyan]Guild Master is synthesizing results...[/bold cyan]"))
    quest_outputs = {qid: qr.output for qid, qr in execution_result.results.items() if qr.success}
    answer = gm.synthesize(courtyard, quest_outputs)

    # Print answer
    console.print()
    console.print(Panel(answer, title="Answer", border_style="green"))

    # Print details
    console.print()
    total_ms = execution_result.execution_time_ms
    detail_table = Table(title="Quest Execution Details", show_header=True)
    detail_table.add_column("Quest")
    detail_table.add_column("Provider")
    detail_table.add_column("Model")
    detail_table.add_column("Latency")
    detail_table.add_column("Status")

    for quest in dag.quests:
        qr = execution_result.results.get(quest.id)
        if qr:
            status_str = "[green]OK[/green]" if qr.success else "[red]FAIL[/red]"
            detail_table.add_row(
                quest.id, qr.provider, qr.model,
                f"{qr.latency_ms}ms", status_str,
            )

    console.print(detail_table)
    console.print(f"\n[dim]Total execution time: {total_ms}ms[/dim]")


def main():
    args = parse_args()

    if args.command == "setup":
        from .wizard import run_wizard
        run_wizard()
        return

    if not args.prompt:
        console.print("[red]Please provide a prompt.[/red]")
        console.print("Usage: tavern \"Your question here\"")
        console.print("       tavern setup")
        sys.exit(1)

    run_quest(args.prompt, model=args.model)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: 4 passed

- [ ] **Step 5: Verify entry point works**

```bash
cd ~/tavern
.venv/bin/python -m tavern --help
```

Expected: Shows usage/help text

- [ ] **Step 6: Commit**

```bash
git add src/tavern/cli.py tests/test_cli.py
git commit -m "feat: CLI module — rich live quest board, arg parsing, main orchestration flow"
```

---

### Task 15: README

**Files:**
- Create: `~/tavern/README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# Tavern

A standalone Python CLI that orchestrates complex prompts across multiple free LLM providers. Submit a prompt, and the **Guild Master** (Claude) decomposes it into a DAG of quests — each with curated context, routed to the best **Adventurer** (provider) for the job — executes them in parallel, and synthesizes a unified answer.

## How It Works

```
tavern "Compare REST vs GraphQL for a mobile app backend"

  Guild Master decomposes into quests:
  ┌───────┬──────────────────────────────┬────────────┬──────────┐
  │ Quest │ Task                         │ Category   │ Provider │
  ├───────┼──────────────────────────────┼────────────┼──────────┤
  │ q1    │ REST pros/cons for mobile    │ speed      │ Groq     │
  │ q2    │ GraphQL pros/cons for mobile │ speed      │ Groq     │
  │ q3    │ Real-world case studies      │ research   │ Gemini   │
  │ q4    │ Comparison table (q1-q3)     │ reasoning  │ OpenRouter│
  └───────┴──────────────────────────────┴────────────┴──────────┘
  q1, q2, q3 run in parallel → q4 runs after → Guild Master synthesizes
```

## Key Feature: Context Curation

Unlike other multi-LLM tools that send the full prompt to every provider, Tavern's Guild Master **curates the context** each Adventurer receives. A security review quest gets only the auth-related code. A documentation quest gets the full file. Smaller, focused prompts = faster inference and better results.

## Adventurers (Free Providers)

| Adventurer | Strength | Free Tier |
|-----------|----------|-----------|
| Groq | Fastest inference (~500 tok/s) | 30 RPM |
| Cerebras | Ultra-fast on small models | ~1M tokens/day |
| SambaNova | Largest open model (405B) | ~200K tokens/day |
| Gemini | 1M context, multimodal | 1,500 req/day |
| OpenRouter | DeepSeek R1 (top reasoning) | Free :free models |
| Mistral | Best free code model (Codestral) | ~1B tokens/month |
| GitHub Models | Free GPT-4o | ~150 req/day |
| Zhipu | GLM-4-Flash, 203K context | ~1M tokens/day |
| Ollama | Offline, unlimited | Local |

## Setup

### 1. Install

```bash
cd ~/tavern
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
tavern setup
```

The wizard prompts for your Anthropic API key (Guild Master) and each Adventurer's key. It validates each with a live test call. Skip any provider to exclude it.

You can also set `ANTHROPIC_API_KEY` as an environment variable.

### 3. Use

```bash
# Basic prompt
tavern "Explain the CAP theorem with examples"

# Pipe context
cat mycode.py | tavern "Review for security issues and suggest improvements"

# Override Guild Master model
tavern --model claude-opus-4-0-20250514 "Deep analysis of distributed consensus"
```

## Task Routing

Each quest is categorized, and providers are ranked per category:

| Category | Best For | Top Pick |
|----------|----------|----------|
| `speed` | Quick lookups, translations | Groq |
| `code` | Code generation, debugging | Mistral (Codestral) |
| `research` | Web-grounded, current info | Gemini |
| `large_context` | Long docs, multimodal | Gemini |
| `reasoning` | Analysis, comparison, logic | OpenRouter (DeepSeek R1) |
| `creative` | Writing, brainstorming | GitHub Models (GPT-4o) |

If the top provider fails, execution automatically falls to the next in the ranking.

## Architecture

```
User prompt + stdin
       │
       ▼
  Courtyard (shared context)
       │
       ▼
  Guild Master (Claude API call #1: Decompose)
  → Produces quest DAG with curated context per quest
       │
       ▼
  Quest Board (shown in terminal, live updates)
       │
       ▼
  Executor (parallel waves, ThreadPoolExecutor)
  → Each Adventurer gets only its curated context
       │
       ▼
  Guild Master (Claude API call #2: Synthesize)
  → Unified answer
```

## License

MIT
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, usage, architecture, and provider details"
```

---

### Task 16: Integration Test

**Files:**
- Create: `tests/test_integration.py`

End-to-end test with mocked providers and mocked Anthropic API.

- [ ] **Step 1: Write the test**

```python
# tests/test_integration.py
import json
from unittest.mock import patch, MagicMock

from tavern.courtyard import Courtyard
from tavern.dag import QuestDAG, Category, Priority
from tavern.executor import DagExecutor
from tavern.guildmaster import GuildMaster, parse_dag_response
from tavern.providers.base import ProviderResult


MOCK_DAG = {
    "prompt": "Compare REST vs GraphQL",
    "quests": [
        {"id": "q1", "task": "REST pros", "context": "Focus on mobile", "category": "speed", "priority": "high", "depends_on": []},
        {"id": "q2", "task": "GraphQL pros", "context": "Focus on mobile", "category": "speed", "priority": "high", "depends_on": []},
        {"id": "q3", "task": "Compare {q1} and {q2}", "context": "", "category": "reasoning", "priority": "critical", "depends_on": ["q1", "q2"]},
    ],
}


class TestEndToEnd:
    def test_full_flow(self):
        # 1. Parse DAG
        dag = parse_dag_response(json.dumps(MOCK_DAG))
        assert len(dag.quests) == 3

        # 2. Execute with mock providers
        def make_mock(name, output):
            m = MagicMock()
            m.name = name
            m.complete.return_value = ProviderResult(
                output=output, success=True, provider=name, model="test", latency_ms=50,
            )
            return m

        providers = {
            "groq": make_mock("groq", "REST is simple and cacheable"),
            "openrouter": make_mock("openrouter", "REST wins for caching, GraphQL for flexibility"),
        }

        executor = DagExecutor(dag, providers, ["groq", "openrouter"])
        result = executor.execute()

        assert result.results["q1"].success
        assert result.results["q2"].success
        assert result.results["q3"].success
        assert result.execution_time_ms > 0
```

- [ ] **Step 2: Run all tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test with mocked providers"
```

---

### Task 17: Final Verification and Push

- [ ] **Step 1: Run full test suite**

```bash
cd ~/tavern
.venv/bin/pytest tests/ -v --tb=short
```

Expected: All tests pass

- [ ] **Step 2: Verify CLI help works**

```bash
.venv/bin/python -m tavern --help
```

- [ ] **Step 3: Verify setup subcommand is recognized**

```bash
.venv/bin/python -m tavern setup --help
```

- [ ] **Step 4: Create GitHub repo and push**

```bash
cd ~/tavern
gh repo create NakulKhanna2001/tavern --public --source=. --push
```

- [ ] **Step 5: Final commit if any loose files**

```bash
git status
```
