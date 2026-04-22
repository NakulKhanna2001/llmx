# llmx — Multi-LLM Orchestrator Plugin for Claude Code

## Goal

A Claude Code plugin that decomposes complex prompts into a DAG of subtasks, routes each to the best-suited LLM provider based on task category, executes independent subtasks in parallel, and synthesizes results with quality oversight.

## Architecture

Hybrid skill + MCP server. A skill definition (`skill.md`) handles the `/llmx` slash command and instructs Claude on decomposition and synthesis. A Python MCP server handles provider communication, parallel execution, setup wizard, and fallback routing. Claude never leaves the session — it orchestrates, the MCP server executes.

## Tech Stack

- Python 3.11+
- httpx — all provider HTTP calls (no SDKs)
- ThreadPoolExecutor — parallel node execution
- rich — CLI output and setup wizard
- Pydantic — DAG schema validation
- PyYAML — config file
- MCP SDK — server protocol

---

## Project Structure

```
~/llmx/
├── pyproject.toml
├── README.md
├── src/
│   └── llmx/
│       ├── __init__.py
│       ├── server.py           # MCP server, exposes tools
│       ├── config.py           # Load/save ~/.llmx/config.yaml
│       ├── wizard.py           # Interactive setup wizard (rich)
│       ├── dag.py              # DAG schema (Pydantic), parsing, validation
│       ├── executor.py         # ThreadPoolExecutor, dependency resolution
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py         # Base provider interface
│       │   ├── groq.py
│       │   ├── openai.py
│       │   ├── gemini.py
│       │   ├── openrouter.py
│       │   └── ollama.py
│       └── fallback.py         # Category-based ranking + fallback logic
├── skill.md                    # Claude Code skill definition
└── claude-plugin.json          # Plugin manifest
```

---

## Data Flow

1. User types `/llmx "prompt"` in Claude Code
2. Skill instructs Claude to call `llmx_status` to check available providers
3. If no providers configured, Claude calls `llmx_setup` (interactive wizard)
4. Claude decomposes the prompt into a DAG (JSON) — each node has an `id`, `task`, `category`, and `depends_on`
5. Claude calls `llmx_preview_dag(dag)` — MCP server validates and returns a formatted table
6. Claude shows the DAG preview to the user, asks for confirmation
7. User confirms — Claude calls `llmx_execute_dag(dag)`
8. MCP server executes nodes in parallel waves using ThreadPoolExecutor
9. Results return to Claude — outputs, provider used, latencies, retries
10. Claude synthesizes a final response; if any node output is poor, calls `llmx_retry_node`
11. Final answer shown prominently, per-node details in collapsible section

---

## DAG Schema

```json
{
  "id": "dag_001",
  "prompt": "Original user prompt",
  "nodes": [
    {
      "id": "n1",
      "task": "Description of what this node should do",
      "category": "speed",
      "depends_on": []
    },
    {
      "id": "n2",
      "task": "Task that needs n1's output. Context from n1: {n1}",
      "category": "code",
      "depends_on": ["n1"]
    }
  ]
}
```

**Rules:**
- `depends_on: []` — node can run immediately
- `{n1}` placeholders in `task` get replaced with that node's output before execution
- `category` is one of: `speed`, `code`, `research`, `large_context`, `reasoning`, `creative`
- No circular dependencies allowed
- All referenced node IDs must exist in the DAG

---

## Task Categories and Provider Rankings

Each category has a ranked list of providers. The executor tries the top available provider first and falls down on failure.

| Category | 1st | 2nd | 3rd | 4th | 5th |
|----------|-----|-----|-----|-----|-----|
| `speed` — simple lookups, fast tasks | Groq | OpenAI | Gemini | OpenRouter | Ollama |
| `code` — generation, debugging | OpenRouter (DeepSeek) | OpenAI | Groq | Gemini | Ollama |
| `large_context` — long docs, multimodal | Gemini | OpenAI | OpenRouter | Groq | Ollama |
| `research` — web-grounded, current info | OpenRouter (Perplexity) | Gemini | OpenAI | Groq | Ollama |
| `reasoning` — analysis, comparison | OpenAI | Gemini | Groq | OpenRouter | Ollama |
| `creative` — writing, brainstorming | OpenAI | Gemini | OpenRouter | Groq | Ollama |

Providers without configured API keys are skipped in the ranking.

**Default models per provider:**
- Groq: `llama-3.1-70b-versatile`
- OpenAI: `gpt-4o`
- Gemini: `gemini-2.0-flash`
- OpenRouter (code): `deepseek/deepseek-coder`
- OpenRouter (research): `perplexity/sonar`
- Ollama: first available model from `api/tags`

---

## Executor

**Wave-based parallel execution:**
1. Sort nodes into waves by dependency depth
2. Wave 1: all nodes with `depends_on: []` — run simultaneously via ThreadPoolExecutor
3. Wave 2: nodes whose dependencies are all in Wave 1 — run after Wave 1 completes
4. Continue until all nodes resolved

**Configuration:**
- `ThreadPoolExecutor(max_workers=8)`
- 30-second timeout per provider call
- Each node runs in its own thread
- Shared dict stores completed outputs
- Dependent nodes block via `threading.Event` until parent nodes complete

**Per-node execution:**
1. Resolve category → ranked provider list
2. Replace `{nX}` placeholders with parent node outputs
3. Try top-ranked available provider
4. On error/429/timeout → try next rank
5. If all ranks exhausted → return with `fallback_exhausted: true` flag

**Return format:**
```json
{
  "results": {
    "n1": {"output": "...", "provider": "groq", "model": "llama-3.1-70b", "latency_ms": 340},
    "n2": {"output": "...", "provider": "openai", "model": "gpt-4o", "latency_ms": 2100}
  },
  "execution_time_ms": 2500,
  "retries": {"n1": ["groq:429 → openai:success"]}
}
```

---

## Quality Check (Claude at Synthesis)

Quality is NOT checked per-node during execution (that would make Claude a bottleneck). Instead:

1. All nodes execute in parallel with mechanical fallback on errors
2. Full results return to Claude
3. Claude synthesizes — if it spots a low-quality output, it calls `llmx_retry_node(node_id)` to re-execute that node with the next provider in ranking
4. Re-synthesis with the improved result
5. If a node has `fallback_exhausted: true`, Claude executes it directly in-session

This keeps the executor fast and Claude as the quality brain without blocking parallelism.

---

## Setup Wizard

**Triggered by:** `llmx_setup` tool or auto on first run when `~/.llmx/config.yaml` is missing.

**Flow:**
1. Prompt for each provider key one by one using rich styled prompts
2. Each key is optional — user can press Enter to skip
3. After each key entry, immediately validate with a lightweight API call
4. Report success/failure for each
5. Ollama: ping `localhost:11434`, list available models
6. Save to `~/.llmx/config.yaml`

**Validation endpoints:**
- Groq: `POST /openai/v1/chat/completions` (1-token prompt)
- OpenAI: `POST /v1/chat/completions` (1-token prompt)
- Gemini: `POST /v1beta/models/gemini-pro:generateContent` (1-token prompt)
- OpenRouter: `GET /api/v1/auth/key` (key info, no generation)
- Ollama: `GET http://localhost:11434/api/tags`

**Config file (`~/.llmx/config.yaml`):**
```yaml
providers:
  groq:
    api_key: "gsk_..."
    validated: true
  openai:
    api_key: "sk-..."
    validated: true
  gemini:
    api_key: null
    validated: false
  openrouter:
    api_key: "sk-or-..."
    validated: true
  ollama:
    available: true
    models: ["llama3", "codellama"]
```

---

## MCP Tools

| Tool | Input | Output |
|------|-------|--------|
| `llmx_setup` | none | Runs interactive wizard, returns config summary |
| `llmx_status` | none | Returns which providers are active and their default models |
| `llmx_preview_dag` | `{dag: DAG JSON}` | Validates schema, returns formatted preview string |
| `llmx_execute_dag` | `{dag: DAG JSON}` | Executes all nodes, returns results + metadata |
| `llmx_retry_node` | `{dag_id, node_id, skip_providers: [...]}` | Re-executes a single node, skipping already-tried providers |

---

## Plugin Manifest (`claude-plugin.json`)

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

---

## v1 Scope

**Included:**
- Setup wizard with key validation
- DAG decomposition + preview + user confirmation
- Parallel execution with ThreadPoolExecutor
- 5 providers: Groq, OpenAI, Gemini, OpenRouter, Ollama
- Category-based routing with ranked fallback
- Claude quality check at synthesis
- Collapsible per-node detail in output
- `llmx_retry_node` for quality re-execution

**Deferred to v2:**
- Piped file/stdin input factored into decomposition
- Streaming output during execution
- DAG caching / history
- Custom ranking overrides in config
- Cost tracking per provider
