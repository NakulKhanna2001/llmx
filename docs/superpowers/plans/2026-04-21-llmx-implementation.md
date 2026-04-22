# llmx Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Claude Code MCP server plugin that decomposes prompts into a DAG of subtasks, routes each to the best-suited LLM provider, executes in parallel, and synthesizes results.

**Architecture:** Hybrid skill + MCP server. `skill.md` instructs Claude on decomposition/synthesis. A Python MCP server (`src/llmx/server.py`) exposes tools for setup, DAG preview, execution, and retry. Providers are httpx-based, execution uses ThreadPoolExecutor.

**Tech Stack:** Python 3.11+, httpx, rich, Pydantic, PyYAML, mcp SDK, ThreadPoolExecutor, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Package metadata, dependencies, entry points |
| `src/llmx/__init__.py` | Package version |
| `src/llmx/config.py` | Load/save `~/.llmx/config.yaml`, provider availability check |
| `src/llmx/wizard.py` | Interactive setup wizard with rich prompts and key validation |
| `src/llmx/dag.py` | Pydantic models for DAG/Node, validation (cycles, refs), preview formatting |
| `src/llmx/providers/base.py` | Abstract base class for providers |
| `src/llmx/providers/groq.py` | Groq httpx provider |
| `src/llmx/providers/openai.py` | OpenAI httpx provider |
| `src/llmx/providers/gemini.py` | Gemini httpx provider |
| `src/llmx/providers/openrouter.py` | OpenRouter httpx provider |
| `src/llmx/providers/ollama.py` | Ollama httpx provider |
| `src/llmx/providers/__init__.py` | Provider registry: name → class mapping |
| `src/llmx/fallback.py` | Category→provider ranking table, resolve available chain |
| `src/llmx/executor.py` | Wave-based parallel executor with ThreadPoolExecutor |
| `src/llmx/server.py` | MCP server exposing all 5 tools |
| `skill.md` | Claude Code skill definition for `/llmx` |
| `claude-plugin.json` | Plugin manifest |
| `tests/test_config.py` | Config load/save tests |
| `tests/test_dag.py` | DAG validation tests |
| `tests/test_fallback.py` | Ranking resolution tests |
| `tests/test_executor.py` | Executor wave ordering and parallel tests |
| `tests/test_providers.py` | Provider call format tests (mocked httpx) |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/llmx/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llmx"
version = "0.1.0"
description = "Multi-LLM orchestrator plugin for Claude Code"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "rich>=13.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "mcp>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
]

[tool.hatch.build.targets.wheel]
packages = ["src/llmx"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `src/llmx/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Create empty test directory**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 4: Install in dev mode and verify**

Run: `cd ~/llmx && pip install -e ".[dev]"`
Expected: Installs successfully, `python -c "import llmx; print(llmx.__version__)"` prints `0.1.0`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: project scaffolding with pyproject.toml"
```

---

### Task 2: Config Module

**Files:**
- Create: `src/llmx/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
import os
import tempfile
from pathlib import Path

from llmx.config import LlmxConfig, load_config, save_config


def test_load_config_missing_file_returns_empty():
    config = load_config(Path("/nonexistent/config.yaml"))
    assert config.providers == {}


def test_save_and_load_roundtrip(tmp_path):
    config_path = tmp_path / "config.yaml"
    config = LlmxConfig(providers={
        "groq": {"api_key": "gsk_test", "validated": True},
        "ollama": {"available": True, "models": ["llama3"]},
    })
    save_config(config, config_path)
    loaded = load_config(config_path)
    assert loaded.providers["groq"]["api_key"] == "gsk_test"
    assert loaded.providers["groq"]["validated"] is True
    assert loaded.providers["ollama"]["models"] == ["llama3"]


def test_get_available_providers():
    config = LlmxConfig(providers={
        "groq": {"api_key": "gsk_test", "validated": True},
        "openai": {"api_key": None, "validated": False},
        "ollama": {"available": True, "models": ["llama3"]},
    })
    available = config.get_available_providers()
    assert "groq" in available
    assert "ollama" in available
    assert "openai" not in available


def test_default_config_path():
    from llmx.config import DEFAULT_CONFIG_PATH
    assert DEFAULT_CONFIG_PATH == Path.home() / ".llmx" / "config.yaml"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/llmx && pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llmx.config'`

- [ ] **Step 3: Implement config module**

```python
# src/llmx/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path.home() / ".llmx" / "config.yaml"


@dataclass
class LlmxConfig:
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_available_providers(self) -> list[str]:
        available = []
        for name, info in self.providers.items():
            if name == "ollama":
                if info.get("available", False):
                    available.append(name)
            elif info.get("validated", False) and info.get("api_key"):
                available.append(name)
        return available


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> LlmxConfig:
    if not path.exists():
        return LlmxConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return LlmxConfig(providers=data.get("providers", {}))


def save_config(config: LlmxConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"providers": config.providers}, f, default_flow_style=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/llmx && pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd ~/llmx && git add src/llmx/config.py tests/test_config.py
git commit -m "feat: config module — load/save ~/.llmx/config.yaml"
```

---

### Task 3: DAG Schema and Validation

**Files:**
- Create: `src/llmx/dag.py`
- Create: `tests/test_dag.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dag.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/llmx && pytest tests/test_dag.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement DAG module**

```python
# src/llmx/dag.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/llmx && pytest tests/test_dag.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
cd ~/llmx && git add src/llmx/dag.py tests/test_dag.py
git commit -m "feat: DAG schema with Pydantic, validation, wave computation"
```

---

### Task 4: Provider Base Class and Groq Provider

**Files:**
- Create: `src/llmx/providers/__init__.py`
- Create: `src/llmx/providers/base.py`
- Create: `src/llmx/providers/groq.py`
- Create: `tests/test_providers.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_providers.py
import httpx
import pytest
import respx

from llmx.providers.base import ProviderResult
from llmx.providers.groq import GroqProvider


@respx.mock
def test_groq_complete_success():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "Hello from Groq"}}],
            "usage": {"total_tokens": 10},
        })
    )
    provider = GroqProvider(api_key="gsk_test")
    result = provider.complete("Say hello")
    assert result.output == "Hello from Groq"
    assert result.success is True


@respx.mock
def test_groq_complete_rate_limit():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    provider = GroqProvider(api_key="gsk_test")
    result = provider.complete("Say hello")
    assert result.success is False
    assert result.error_code == 429


@respx.mock
def test_groq_validate_success():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 1},
        })
    )
    provider = GroqProvider(api_key="gsk_test")
    assert provider.validate() is True


@respx.mock
def test_groq_validate_bad_key():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "invalid key"}})
    )
    provider = GroqProvider(api_key="bad_key")
    assert provider.validate() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/llmx && pytest tests/test_providers.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement base class**

```python
# src/llmx/providers/base.py
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

- [ ] **Step 4: Implement Groq provider**

```python
# src/llmx/providers/groq.py
import time

import httpx

from .base import BaseProvider, ProviderResult

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(BaseProvider):
    name = "groq"
    default_model = "llama-3.1-70b-versatile"

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
                    output="",
                    success=False,
                    provider=self.name,
                    model=model,
                    latency_ms=latency,
                    error_code=resp.status_code,
                    error_message=resp.text,
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return ProviderResult(
                output=content,
                success=True,
                provider=self.name,
                model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="",
                success=False,
                provider=self.name,
                model=model,
                latency_ms=latency,
                error_code=None,
                error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        result = self.complete("Hi", model=self.default_model)
        return result.success
```

- [ ] **Step 5: Create providers `__init__.py`**

```python
# src/llmx/providers/__init__.py
from .groq import GroqProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqProvider,
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd ~/llmx && pytest tests/test_providers.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
cd ~/llmx && git add src/llmx/providers/ tests/test_providers.py
git commit -m "feat: provider base class and Groq provider with httpx"
```

---

### Task 5: OpenAI Provider

**Files:**
- Create: `src/llmx/providers/openai.py`
- Modify: `src/llmx/providers/__init__.py`
- Modify: `tests/test_providers.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_providers.py`:

```python
from llmx.providers.openai import OpenAIProvider


@respx.mock
def test_openai_complete_success():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "Hello from OpenAI"}}],
            "usage": {"total_tokens": 10},
        })
    )
    provider = OpenAIProvider(api_key="sk-test")
    result = provider.complete("Say hello")
    assert result.output == "Hello from OpenAI"
    assert result.success is True


@respx.mock
def test_openai_complete_rate_limit():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    provider = OpenAIProvider(api_key="sk-test")
    result = provider.complete("Say hello")
    assert result.success is False
    assert result.error_code == 429
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/llmx && pytest tests/test_providers.py::test_openai_complete_success -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement OpenAI provider**

```python
# src/llmx/providers/openai.py
import time

import httpx

from .base import BaseProvider, ProviderResult

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIProvider(BaseProvider):
    name = "openai"
    default_model = "gpt-4o"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        start = time.monotonic()
        try:
            resp = httpx.post(
                OPENAI_API_URL,
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
                    output="",
                    success=False,
                    provider=self.name,
                    model=model,
                    latency_ms=latency,
                    error_code=resp.status_code,
                    error_message=resp.text,
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return ProviderResult(
                output=content,
                success=True,
                provider=self.name,
                model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="",
                success=False,
                provider=self.name,
                model=model,
                latency_ms=latency,
                error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        result = self.complete("Hi", model=self.default_model)
        return result.success
```

- [ ] **Step 4: Update provider registry**

```python
# src/llmx/providers/__init__.py
from .groq import GroqProvider
from .openai import OpenAIProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/llmx && pytest tests/test_providers.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
cd ~/llmx && git add src/llmx/providers/openai.py src/llmx/providers/__init__.py tests/test_providers.py
git commit -m "feat: OpenAI provider with httpx"
```

---

### Task 6: Gemini Provider

**Files:**
- Create: `src/llmx/providers/gemini.py`
- Modify: `src/llmx/providers/__init__.py`
- Modify: `tests/test_providers.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_providers.py`:

```python
from llmx.providers.gemini import GeminiProvider


@respx.mock
def test_gemini_complete_success():
    respx.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent").mock(
        return_value=httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "Hello from Gemini"}]}}],
        })
    )
    provider = GeminiProvider(api_key="gem_test")
    result = provider.complete("Say hello")
    assert result.output == "Hello from Gemini"
    assert result.success is True


@respx.mock
def test_gemini_complete_rate_limit():
    respx.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    provider = GeminiProvider(api_key="gem_test")
    result = provider.complete("Say hello")
    assert result.success is False
    assert result.error_code == 429
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/llmx && pytest tests/test_providers.py::test_gemini_complete_success -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Gemini provider**

```python
# src/llmx/providers/gemini.py
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
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                },
                timeout=30.0,
            )
            latency = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return ProviderResult(
                    output="",
                    success=False,
                    provider=self.name,
                    model=model,
                    latency_ms=latency,
                    error_code=resp.status_code,
                    error_message=resp.text,
                )

            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            return ProviderResult(
                output=content,
                success=True,
                provider=self.name,
                model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="",
                success=False,
                provider=self.name,
                model=model,
                latency_ms=latency,
                error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        result = self.complete("Hi", model=self.default_model)
        return result.success
```

- [ ] **Step 4: Update provider registry**

```python
# src/llmx/providers/__init__.py
from .groq import GroqProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/llmx && pytest tests/test_providers.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
cd ~/llmx && git add src/llmx/providers/gemini.py src/llmx/providers/__init__.py tests/test_providers.py
git commit -m "feat: Gemini provider with httpx"
```

---

### Task 7: OpenRouter Provider

**Files:**
- Create: `src/llmx/providers/openrouter.py`
- Modify: `src/llmx/providers/__init__.py`
- Modify: `tests/test_providers.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_providers.py`:

```python
from llmx.providers.openrouter import OpenRouterProvider


@respx.mock
def test_openrouter_complete_success():
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "Hello from OpenRouter"}}],
            "usage": {"total_tokens": 10},
        })
    )
    provider = OpenRouterProvider(api_key="sk-or-test")
    result = provider.complete("Say hello", model="deepseek/deepseek-coder")
    assert result.output == "Hello from OpenRouter"
    assert result.success is True


@respx.mock
def test_openrouter_validate_success():
    respx.get("https://openrouter.ai/api/v1/auth/key").mock(
        return_value=httpx.Response(200, json={"data": {"label": "test"}})
    )
    provider = OpenRouterProvider(api_key="sk-or-test")
    assert provider.validate() is True


@respx.mock
def test_openrouter_validate_bad_key():
    respx.get("https://openrouter.ai/api/v1/auth/key").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    provider = OpenRouterProvider(api_key="bad")
    assert provider.validate() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/llmx && pytest tests/test_providers.py::test_openrouter_complete_success -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement OpenRouter provider**

```python
# src/llmx/providers/openrouter.py
import time

import httpx

from .base import BaseProvider, ProviderResult

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_AUTH_URL = "https://openrouter.ai/api/v1/auth/key"


class OpenRouterProvider(BaseProvider):
    name = "openrouter"
    default_model = "deepseek/deepseek-coder"

    # Category-specific default models
    MODEL_FOR_CATEGORY = {
        "code": "deepseek/deepseek-coder",
        "research": "perplexity/sonar",
    }

    def __init__(self, api_key: str):
        self.api_key = api_key

    def model_for_category(self, category: str) -> str:
        return self.MODEL_FOR_CATEGORY.get(category, self.default_model)

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
                    output="",
                    success=False,
                    provider=self.name,
                    model=model,
                    latency_ms=latency,
                    error_code=resp.status_code,
                    error_message=resp.text,
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return ProviderResult(
                output=content,
                success=True,
                provider=self.name,
                model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="",
                success=False,
                provider=self.name,
                model=model,
                latency_ms=latency,
                error_message="Timeout after 30s",
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

- [ ] **Step 4: Update provider registry**

```python
# src/llmx/providers/__init__.py
from .groq import GroqProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider
from .openrouter import OpenRouterProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/llmx && pytest tests/test_providers.py -v`
Expected: 11 passed

- [ ] **Step 6: Commit**

```bash
cd ~/llmx && git add src/llmx/providers/openrouter.py src/llmx/providers/__init__.py tests/test_providers.py
git commit -m "feat: OpenRouter provider with category-specific model routing"
```

---

### Task 8: Ollama Provider

**Files:**
- Create: `src/llmx/providers/ollama.py`
- Modify: `src/llmx/providers/__init__.py`
- Modify: `tests/test_providers.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_providers.py`:

```python
from llmx.providers.ollama import OllamaProvider


@respx.mock
def test_ollama_complete_success():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json={
            "response": "Hello from Ollama",
            "done": True,
        })
    )
    provider = OllamaProvider(model="llama3")
    result = provider.complete("Say hello")
    assert result.output == "Hello from Ollama"
    assert result.success is True


@respx.mock
def test_ollama_validate_running():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={
            "models": [{"name": "llama3"}, {"name": "codellama"}],
        })
    )
    provider = OllamaProvider()
    is_valid, models = provider.validate_with_models()
    assert is_valid is True
    assert "llama3" in models


@respx.mock
def test_ollama_validate_not_running():
    respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    provider = OllamaProvider()
    is_valid, models = provider.validate_with_models()
    assert is_valid is False
    assert models == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/llmx && pytest tests/test_providers.py::test_ollama_complete_success -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Ollama provider**

```python
# src/llmx/providers/ollama.py
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
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60.0,  # Ollama can be slow
            )
            latency = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return ProviderResult(
                    output="",
                    success=False,
                    provider=self.name,
                    model=model,
                    latency_ms=latency,
                    error_code=resp.status_code,
                    error_message=resp.text,
                )

            data = resp.json()
            return ProviderResult(
                output=data["response"],
                success=True,
                provider=self.name,
                model=model,
                latency_ms=latency,
            )
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="",
                success=False,
                provider=self.name,
                model=model,
                latency_ms=latency,
                error_message=str(e),
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

- [ ] **Step 4: Update provider registry**

```python
# src/llmx/providers/__init__.py
from .groq import GroqProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider
from .openrouter import OpenRouterProvider
from .ollama import OllamaProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
    "ollama": OllamaProvider,
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/llmx && pytest tests/test_providers.py -v`
Expected: 14 passed

- [ ] **Step 6: Commit**

```bash
cd ~/llmx && git add src/llmx/providers/ollama.py src/llmx/providers/__init__.py tests/test_providers.py
git commit -m "feat: Ollama local provider with model discovery"
```

---

### Task 9: Fallback Routing

**Files:**
- Create: `src/llmx/fallback.py`
- Create: `tests/test_fallback.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fallback.py
from llmx.fallback import get_provider_chain, CATEGORY_RANKINGS


def test_all_categories_have_rankings():
    expected = {"speed", "code", "research", "large_context", "reasoning", "creative"}
    assert set(CATEGORY_RANKINGS.keys()) == expected


def test_get_provider_chain_all_available():
    available = ["groq", "openai", "gemini", "openrouter", "ollama"]
    chain = get_provider_chain("speed", available)
    assert chain == ["groq", "openai", "gemini", "openrouter", "ollama"]


def test_get_provider_chain_some_missing():
    available = ["groq", "ollama"]
    chain = get_provider_chain("code", available)
    # code ranking: openrouter, openai, groq, gemini, ollama
    # only groq and ollama available
    assert chain == ["groq", "ollama"]


def test_get_provider_chain_none_available():
    chain = get_provider_chain("speed", [])
    assert chain == []


def test_code_category_prefers_openrouter():
    available = ["groq", "openai", "gemini", "openrouter", "ollama"]
    chain = get_provider_chain("code", available)
    assert chain[0] == "openrouter"


def test_research_category_prefers_openrouter():
    available = ["groq", "openai", "gemini", "openrouter", "ollama"]
    chain = get_provider_chain("research", available)
    assert chain[0] == "openrouter"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/llmx && pytest tests/test_fallback.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement fallback module**

```python
# src/llmx/fallback.py

CATEGORY_RANKINGS: dict[str, list[str]] = {
    "speed":         ["groq", "openai", "gemini", "openrouter", "ollama"],
    "code":          ["openrouter", "openai", "groq", "gemini", "ollama"],
    "large_context": ["gemini", "openai", "openrouter", "groq", "ollama"],
    "research":      ["openrouter", "gemini", "openai", "groq", "ollama"],
    "reasoning":     ["openai", "gemini", "groq", "openrouter", "ollama"],
    "creative":      ["openai", "gemini", "openrouter", "groq", "ollama"],
}

# Category-specific models for OpenRouter
OPENROUTER_CATEGORY_MODELS: dict[str, str] = {
    "code": "deepseek/deepseek-coder",
    "research": "perplexity/sonar",
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
    if provider == "openrouter":
        return OPENROUTER_CATEGORY_MODELS.get(category)
    return None  # use provider default
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/llmx && pytest tests/test_fallback.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
cd ~/llmx && git add src/llmx/fallback.py tests/test_fallback.py
git commit -m "feat: category-based provider ranking and fallback chains"
```

---

### Task 10: DAG Executor

**Files:**
- Create: `src/llmx/executor.py`
- Create: `tests/test_executor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_executor.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/llmx && pytest tests/test_executor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement executor**

```python
# src/llmx/executor.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/llmx && pytest tests/test_executor.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
cd ~/llmx && git add src/llmx/executor.py tests/test_executor.py
git commit -m "feat: wave-based parallel DAG executor with fallback"
```

---

### Task 11: Setup Wizard

**Files:**
- Create: `src/llmx/wizard.py`

- [ ] **Step 1: Implement wizard**

```python
# src/llmx/wizard.py
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

from .config import LlmxConfig, save_config, DEFAULT_CONFIG_PATH
from .providers.groq import GroqProvider
from .providers.openai import OpenAIProvider
from .providers.gemini import GeminiProvider
from .providers.openrouter import OpenRouterProvider
from .providers.ollama import OllamaProvider

console = Console()

PROVIDER_INFO = [
    {
        "name": "groq",
        "label": "Groq",
        "description": "Fast inference for simple tasks",
        "url": "https://console.groq.com/keys",
        "key_prefix": "gsk_",
    },
    {
        "name": "openai",
        "label": "OpenAI",
        "description": "General reasoning, creative writing",
        "url": "https://platform.openai.com/api-keys",
        "key_prefix": "sk-",
    },
    {
        "name": "gemini",
        "label": "Google Gemini",
        "description": "Large context, multimodal",
        "url": "https://aistudio.google.com/apikey",
        "key_prefix": "",
    },
    {
        "name": "openrouter",
        "label": "OpenRouter",
        "description": "DeepSeek (code), Perplexity (research)",
        "url": "https://openrouter.ai/keys",
        "key_prefix": "sk-or-",
    },
]


def run_wizard() -> dict:
    console.print(Panel(
        Text("llmx Setup Wizard", style="bold cyan", justify="center"),
        subtitle="Configure your LLM providers",
    ))
    console.print()

    config = LlmxConfig()
    total = len(PROVIDER_INFO) + 1  # +1 for Ollama
    active_count = 0

    for i, info in enumerate(PROVIDER_INFO, 1):
        console.print(f"[bold][{i}/{total}] {info['label']}[/bold] — {info['description']}")
        console.print(f"       Get a key at: [link]{info['url']}[/link]")

        key = Prompt.ask(
            "       Enter key (or press Enter to skip)",
            default="",
            show_default=False,
        )

        if not key:
            console.print("       [dim]Skipped[/dim]")
            config.providers[info["name"]] = {"api_key": None, "validated": False}
        else:
            console.print("       Validating...", end=" ")
            provider_cls = {
                "groq": GroqProvider,
                "openai": OpenAIProvider,
                "gemini": GeminiProvider,
                "openrouter": OpenRouterProvider,
            }[info["name"]]
            provider = provider_cls(api_key=key)
            valid = provider.validate()

            if valid:
                console.print("[green]Validated[/green]")
                config.providers[info["name"]] = {"api_key": key, "validated": True}
                active_count += 1
            else:
                console.print("[red]Failed[/red] — key will not be used")
                config.providers[info["name"]] = {"api_key": key, "validated": False}

        console.print()

    # Ollama
    console.print(f"[bold][{total}/{total}] Ollama[/bold] — Local fallback, no key needed")
    console.print("       Checking localhost:11434...", end=" ")
    ollama = OllamaProvider()
    is_valid, models = ollama.validate_with_models()

    if is_valid:
        console.print(f"[green]Running[/green] — models: {', '.join(models)}")
        config.providers["ollama"] = {"available": True, "models": models}
        active_count += 1
    else:
        console.print("[yellow]Not running[/yellow]")
        config.providers["ollama"] = {"available": False, "models": []}

    console.print()
    save_config(config)
    console.print(Panel(
        f"[bold green]{active_count}/{total} providers active[/bold green]\n"
        f"Config saved to {DEFAULT_CONFIG_PATH}",
        title="Setup Complete",
    ))

    return {
        "active_providers": config.get_available_providers(),
        "config_path": str(DEFAULT_CONFIG_PATH),
    }
```

- [ ] **Step 2: Verify import works**

Run: `cd ~/llmx && python -c "from llmx.wizard import run_wizard; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
cd ~/llmx && git add src/llmx/wizard.py
git commit -m "feat: interactive setup wizard with rich prompts and key validation"
```

---

### Task 12: MCP Server

**Files:**
- Create: `src/llmx/server.py`

- [ ] **Step 1: Implement MCP server**

```python
# src/llmx/server.py
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .config import load_config, DEFAULT_CONFIG_PATH
from .dag import Dag, validate_dag, format_dag_preview, DagValidationError
from .executor import DagExecutor
from .fallback import get_provider_chain
from .providers import PROVIDER_REGISTRY
from .providers.base import BaseProvider
from .wizard import run_wizard

server = Server("llmx")

# In-memory store for last execution (for retry)
_last_dag: Dag | None = None
_last_providers: dict[str, BaseProvider] = {}
_last_available: list[str] = []


def _build_providers() -> tuple[dict[str, BaseProvider], list[str]]:
    config = load_config()
    available = config.get_available_providers()
    providers: dict[str, BaseProvider] = {}

    for name in available:
        info = config.providers.get(name, {})
        cls = PROVIDER_REGISTRY.get(name)
        if cls is None:
            continue
        if name == "ollama":
            models = info.get("models", [])
            providers[name] = cls(model=models[0] if models else None)
        else:
            api_key = info.get("api_key", "")
            providers[name] = cls(api_key=api_key)

    return providers, available


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="llmx_setup",
            description="Run the interactive setup wizard to configure LLM provider API keys",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="llmx_status",
            description="Check which LLM providers are configured and available",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="llmx_preview_dag",
            description="Validate a DAG and return a formatted preview for user confirmation",
            inputSchema={
                "type": "object",
                "properties": {
                    "dag": {
                        "type": "object",
                        "description": "The DAG JSON with id, prompt, and nodes",
                    },
                },
                "required": ["dag"],
            },
        ),
        Tool(
            name="llmx_execute_dag",
            description="Execute a validated DAG — runs nodes in parallel across providers, returns all results",
            inputSchema={
                "type": "object",
                "properties": {
                    "dag": {
                        "type": "object",
                        "description": "The DAG JSON with id, prompt, and nodes",
                    },
                },
                "required": ["dag"],
            },
        ),
        Tool(
            name="llmx_retry_node",
            description="Re-execute a single node, skipping providers that already failed",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "skip_providers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Providers to skip (already tried)",
                    },
                },
                "required": ["node_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    global _last_dag, _last_providers, _last_available

    if name == "llmx_setup":
        result = run_wizard()
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "llmx_status":
        config = load_config()
        available = config.get_available_providers()
        status = {
            "configured": available,
            "missing": [p for p in PROVIDER_REGISTRY if p not in available],
            "config_path": str(DEFAULT_CONFIG_PATH),
        }
        return [TextContent(type="text", text=json.dumps(status, indent=2))]

    if name == "llmx_preview_dag":
        try:
            dag = Dag(**arguments["dag"])
            validate_dag(dag)
            preview = format_dag_preview(dag)
            return [TextContent(type="text", text=preview)]
        except (DagValidationError, Exception) as e:
            return [TextContent(type="text", text=f"DAG validation error: {e}")]

    if name == "llmx_execute_dag":
        try:
            dag = Dag(**arguments["dag"])
            validate_dag(dag)
            providers, available = _build_providers()

            _last_dag = dag
            _last_providers = providers
            _last_available = available

            executor = DagExecutor(dag, providers, available)
            result = executor.execute()

            output = {
                "results": {},
                "execution_time_ms": result.execution_time_ms,
                "retries": result.retries,
            }
            for node_id, node_result in result.results.items():
                output["results"][node_id] = {
                    "output": node_result.output,
                    "success": node_result.success,
                    "provider": node_result.provider,
                    "model": node_result.model,
                    "latency_ms": node_result.latency_ms,
                    "fallback_exhausted": node_result.fallback_exhausted,
                }

            return [TextContent(type="text", text=json.dumps(output, indent=2))]
        except (DagValidationError, Exception) as e:
            return [TextContent(type="text", text=f"Execution error: {e}")]

    if name == "llmx_retry_node":
        node_id = arguments["node_id"]
        skip = arguments.get("skip_providers", [])

        if _last_dag is None:
            return [TextContent(type="text", text="No previous DAG execution found")]

        node = next((n for n in _last_dag.nodes if n.id == node_id), None)
        if node is None:
            return [TextContent(type="text", text=f"Node {node_id} not found in last DAG")]

        chain = get_provider_chain(node.category.value, _last_available, skip_providers=skip)
        if not chain:
            return [TextContent(type="text", text=json.dumps({
                "node_id": node_id,
                "success": False,
                "fallback_exhausted": True,
            }))]

        prompt = node.task
        # Substitute from last execution outputs
        import re
        for match in re.finditer(r"\{(n\d+)\}", prompt):
            dep_id = match.group(1)
            # Get from last execution context if available
            prompt = prompt.replace(f"{{{dep_id}}}", "")

        for provider_name in chain:
            provider = _last_providers.get(provider_name)
            if provider is None:
                continue
            result = provider.complete(prompt)
            if result.success:
                return [TextContent(type="text", text=json.dumps({
                    "node_id": node_id,
                    "output": result.output,
                    "success": True,
                    "provider": result.provider,
                    "model": result.model,
                    "latency_ms": result.latency_ms,
                }))]

        return [TextContent(type="text", text=json.dumps({
            "node_id": node_id,
            "success": False,
            "fallback_exhausted": True,
        }))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

- [ ] **Step 2: Verify import works**

Run: `cd ~/llmx && python -c "from llmx.server import server; print('MCP server ok')"`
Expected: `MCP server ok`

- [ ] **Step 3: Commit**

```bash
cd ~/llmx && git add src/llmx/server.py
git commit -m "feat: MCP server exposing setup, status, preview, execute, retry tools"
```

---

### Task 13: Skill Definition and Plugin Manifest

**Files:**
- Create: `skill.md`
- Create: `claude-plugin.json`

- [ ] **Step 1: Create skill definition**

```markdown
<!-- skill.md -->
---
name: llmx
description: Decompose a complex prompt into subtasks, execute across multiple LLM providers in parallel, synthesize results
command: /llmx
---

# llmx — Multi-LLM Orchestrator

You are using the llmx plugin to orchestrate complex prompts across multiple LLM providers.

## Flow

1. **Check providers**: Call `llmx_status` to see which providers are available. If none are configured, call `llmx_setup` first.

2. **Decompose the prompt**: Analyze the user's prompt and break it into a DAG of subtasks. Each node must have:
   - `id`: unique identifier (n1, n2, n3...)
   - `task`: what this node should accomplish. Use `{nX}` to reference another node's output.
   - `category`: one of `speed`, `code`, `research`, `large_context`, `reasoning`, `creative`
   - `depends_on`: list of node IDs this node needs (empty if independent)

   Maximize parallelism — independent subtasks should have no dependencies.

3. **Preview**: Call `llmx_preview_dag` with your DAG JSON. Show the formatted preview to the user and ask them to confirm, edit, or cancel.

4. **Execute**: On confirmation, call `llmx_execute_dag`. This runs all nodes in parallel where possible.

5. **Synthesize**: Review all node results. If any output is low quality or a node has `fallback_exhausted: true`:
   - For low quality: call `llmx_retry_node` with `skip_providers` listing the provider that gave the bad result
   - For `fallback_exhausted`: execute that subtask yourself directly using your own knowledge

6. **Present results**: Give the user a clear, unified answer. After the main response, include a collapsible details section:

   <details>
   <summary>Node execution details</summary>

   For each node: task, provider used, latency, output summary.

   </details>

## Category Guide

- `speed` — simple factual lookups, translations, summaries of short text
- `code` — code generation, debugging, code review, refactoring
- `research` — questions needing current/web-grounded information
- `large_context` — tasks involving long documents or multiple files
- `reasoning` — analysis, comparison, logical deduction, math
- `creative` — writing, brainstorming, ideation, marketing copy
```

- [ ] **Step 2: Create plugin manifest**

```json
{
  "name": "llmx",
  "description": "Multi-LLM orchestrator — decomposes prompts into parallel subtasks across providers",
  "version": "0.1.0",
  "skills": [
    {
      "name": "llmx",
      "description": "Decompose a complex prompt into subtasks, execute across multiple LLM providers in parallel, synthesize results",
      "command": "/llmx"
    }
  ],
  "mcp_server": {
    "command": "python",
    "args": ["-m", "llmx.server"]
  }
}
```

- [ ] **Step 3: Commit**

```bash
cd ~/llmx && git add skill.md claude-plugin.json
git commit -m "feat: Claude Code skill definition and plugin manifest"
```

---

### Task 14: Integration Test — Full DAG Round Trip

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
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
```

- [ ] **Step 2: Run the integration test**

Run: `cd ~/llmx && pytest tests/test_integration.py -v`
Expected: 1 passed

- [ ] **Step 3: Run full test suite**

Run: `cd ~/llmx && pytest -v`
Expected: All tests pass (approximately 20+ tests)

- [ ] **Step 4: Commit**

```bash
cd ~/llmx && git add tests/test_integration.py
git commit -m "test: full DAG round-trip integration test"
```

---

### Task 15: Final Polish — `__main__` and GitHub Repo

**Files:**
- Create: `src/llmx/__main__.py`

- [ ] **Step 1: Create `__main__.py` for `python -m llmx.server`**

```python
# src/llmx/__main__.py
from .server import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

- [ ] **Step 2: Verify the server can start**

Run: `cd ~/llmx && timeout 2 python -m llmx.server 2>&1 || true`
Expected: No import errors (the server will hang waiting for stdio input — that's fine, it just proves it loads)

- [ ] **Step 3: Create GitHub repo and push**

```bash
cd ~/llmx
gh repo create NakulKhanna2001/llmx --public --description "Multi-LLM orchestrator plugin for Claude Code" --source=. --push
```

- [ ] **Step 4: Final commit**

```bash
cd ~/llmx && git add src/llmx/__main__.py
git commit -m "feat: add __main__.py entry point for MCP server"
git push
```
