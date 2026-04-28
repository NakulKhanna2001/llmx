# Tavern — Multi-LLM Orchestrator Design Spec

## Overview

Tavern is a standalone Python CLI that orchestrates complex prompts across multiple free LLM providers. It uses a **Guild metaphor**: a **Guild Master** (Claude via Anthropic API) decomposes a user's prompt into a DAG of quests, curates the exact context each quest needs, and routes them to **Adventurers** (LLM providers) based on task category. Quests execute in parallel where possible, and the Guild Master synthesizes results into a unified answer.

**Key differentiator**: The Guild Master decides *how much context* each adventurer receives — not a fixed prompt copy, but a curated slice tailored to each quest's needs.

**Target**: `~/tavern/` — new standalone project. Reuses proven patterns from the existing `~/llmx/` plugin (DAG schema, executor, provider base class, fallback routing).

---

## 1. Guild Metaphor — Core Concepts

### Courtyard (Shared Context Space)

The Courtyard holds everything the Guild Master can draw from when curating context for quests:

- **User prompt** — the original question/request
- **Piped stdin** — file contents, code, logs piped into tavern (e.g., `cat file.py | tavern "review this"`)

The Courtyard is input-only. Adventurers never see the full Courtyard — they only see the slice the Guild Master curates for their specific quest.

### Bulletin Board (Quest DAG)

The Bulletin Board is the DAG of quests posted by the Guild Master. Each quest node contains:

- `id` — unique identifier (q1, q2, q3...)
- `task` — what the adventurer should accomplish
- `context` — the curated slice of the Courtyard relevant to this quest (Guild Master decides this)
- `category` — one of: `speed`, `code`, `research`, `large_context`, `reasoning`, `creative`
- `priority` — `critical`, `high`, `normal`, or `low`
- `depends_on` — list of quest IDs this quest needs completed first. Use `{qX}` placeholders to reference another quest's output.

### Guild Master (Claude)

Claude (via Anthropic API) serves two roles:

1. **Decompose** — Analyze the Courtyard, break the prompt into a quest DAG, categorize each quest, assign priority, and curate the context slice for each quest
2. **Synthesize** — Review all quest results, handle quality issues, produce the final unified answer

Two API calls total: one to decompose, one to synthesize.

### Adventurers (LLM Providers)

Free-tier LLM providers that execute quests. Each adventurer receives only:
- The quest's `task` description
- The quest's curated `context`
- Any referenced outputs from dependency quests (`{qX}` substitution)

They never see the full Courtyard or other quests.

---

## 2. Task Routing — Category-Based Provider Rankings

The Guild Master categorizes each quest. A static ranking table determines which adventurer gets the quest. If the top-ranked adventurer fails (rate limit, error, timeout), execution falls to the next in line.

### Category Rankings

| Category | 1st Pick | 2nd | 3rd | 4th | Fallback |
|----------|----------|-----|-----|-----|----------|
| `speed` | Groq | Cerebras | SambaNova | Gemini | Ollama |
| `code` | Mistral | OpenRouter | GitHub Models | Cerebras | Ollama |
| `research` | Gemini | OpenRouter | GitHub Models | SambaNova | Ollama |
| `large_context` | Gemini | Zhipu | Mistral | GitHub Models | Ollama |
| `reasoning` | OpenRouter | SambaNova | Cerebras | GitHub Models | Ollama |
| `creative` | GitHub Models | Mistral | SambaNova | Gemini | Ollama |

**Ollama** is the universal last-resort fallback for every category — offline, unlimited, no API key needed. Its quality depends on the user's locally pulled models.

### Category Definitions

| Category | Best For | Examples |
|----------|----------|---------|
| `speed` | Quick lookups, translations, summaries | "Translate this to French", "Summarize in one line" |
| `code` | Code generation, debugging, refactoring | "Write a Python function for X", "Find the bug" |
| `research` | Web-grounded, current information | "Latest stats on X", "Compare recent Y" |
| `large_context` | Long documents, multimodal input | "Analyze this 50-page doc", "Review this codebase" |
| `reasoning` | Analysis, comparison, logic, math | "Compare A vs B", "Prove this theorem" |
| `creative` | Writing, brainstorming, ideation | "Write marketing copy", "Brainstorm names" |

### Priority System

Each quest gets a priority that controls execution order within dependency waves:

| Priority | Meaning | Execution Behavior |
|----------|---------|--------------------|
| `critical` | Must succeed for final answer | Executed first in wave, more retries |
| `high` | Important but not blocking | Executed early in wave |
| `normal` | Standard quest | Default execution order |
| `low` | Nice to have | Executed last, fewer retries |

Within each wave (set of quests whose dependencies are satisfied), higher-priority quests are submitted to the thread pool first.

---

## 3. Context Curation — The Key Differentiator

### How It Works

When the Guild Master decomposes a prompt, it doesn't just split the task — it decides what each adventurer needs to know. For each quest node, the Guild Master produces a `context` field containing only the relevant slice of the Courtyard.

### Example

**Courtyard**: User pipes in a 500-line Python file and asks "Review this code for security issues, suggest performance improvements, and write documentation."

**Guild Master produces**:
- **q1** (security review, category: `reasoning`): `context` = only the functions with user input handling, database calls, auth logic (~80 lines)
- **q2** (performance review, category: `code`): `context` = hot loops, database queries, data structures (~100 lines)
- **q3** (documentation, category: `creative`): `context` = full file (needs to see everything to document it)
- **q4** (synthesis, depends on q1-q3): `context` = empty (receives q1-q3 outputs via `{qX}` placeholders)

### Guild Master Decomposition Prompt

The system prompt sent to Claude for decomposition includes:
- The full Courtyard contents
- Instructions to produce a JSON DAG
- The category definitions and priority levels
- Explicit instruction to curate context per quest — include only what that specific adventurer needs
- The JSON schema for the DAG

### Benefits

- Smaller prompts = faster inference, lower token usage on free tiers
- Adventurers aren't distracted by irrelevant context
- Sensitive sections can be routed only to trusted providers
- Large files don't need to be sent in full to every provider

---

## 4. Adventurer Roster — 9 Free Providers

### Provider Details

| # | Adventurer | Free Tier Limits | Top Models | Strength |
|---|-----------|-----------------|------------|----------|
| 1 | **Groq** | 30 RPM, 14.4K req/day | Llama 3.3 70B, Llama 4 Scout, DeepSeek R1 Distill 70B, Qwen QwQ 32B | Fastest inference (~500 tok/s) |
| 2 | **Cerebras** | ~1M tokens/day | Llama 3.1 8B, Qwen3-235B | Ultra-fast on small models |
| 3 | **SambaNova** | ~200K tokens/day | Llama 3.3 70B, Llama 3.1 405B, DeepSeek R1 Distill 70B | Access to 405B param model |
| 4 | **Gemini** | 1,500 req/day | Gemini 2.5 Pro, Gemini 2.0 Flash | 1M context, multimodal, search grounding |
| 5 | **OpenRouter** | Free `:free` models | DeepSeek R1, Llama 3.3 70B, Gemma 3 12B | Meta-provider, top reasoning model |
| 6 | **Mistral** | ~1B tokens/month | Codestral, Mistral Large, Mistral Small | Best free code model |
| 7 | **GitHub Models** | ~150 req/day | GPT-4o, DeepSeek-R1, Llama 3.3 70B, Mistral Large | Access to GPT-4o for free |
| 8 | **Zhipu** | ~1M tokens/day | GLM-4.7-Flash | 203K context, Chinese AI lab |
| 9 | **Ollama** | Unlimited (local) | User's pulled models | Offline, no API key |

### Default Models per Category

| Category | Groq | Cerebras | SambaNova | Gemini | OpenRouter | Mistral | GitHub | Zhipu |
|----------|------|----------|-----------|--------|------------|---------|--------|-------|
| `speed` | Llama 3.3 70B | Llama 3.1 8B | Llama 3.3 70B | Flash 2.0 | Llama 3.3 70B | Small | Llama 3.3 70B | GLM-4.7-Flash |
| `code` | Qwen QwQ 32B | Qwen3-235B | DeepSeek R1 Distill | Flash 2.0 | DeepSeek R1 | Codestral | GPT-4o | GLM-4.7-Flash |
| `research` | Llama 3.3 70B | Qwen3-235B | Llama 3.1 405B | 2.5 Pro | DeepSeek R1 | Large | GPT-4o | GLM-4.7-Flash |
| `large_context` | Llama 3.3 70B | Qwen3-235B | Llama 3.1 405B | 2.5 Pro | DeepSeek R1 | Large | GPT-4o | GLM-4.7-Flash |
| `reasoning` | Qwen QwQ 32B | Qwen3-235B | Llama 3.1 405B | 2.5 Pro | DeepSeek R1 | Large | GPT-4o | GLM-4.7-Flash |
| `creative` | Llama 3.3 70B | Qwen3-235B | Llama 3.1 405B | 2.5 Pro | Llama 3.3 70B | Large | GPT-4o | GLM-4.7-Flash |

---

## 5. CLI Interface

### Invocation

```bash
# Basic usage
tavern "Compare REST vs GraphQL for a mobile app backend"

# With piped context
cat api.py | tavern "Review this for security issues"

# With Guild Master model override
tavern --model claude-sonnet-4-5-20250514 "Explain quantum computing"
```

### Output — Live Rich Display

```
╔══════════════════════════════════════════════════════╗
║  🏰 Tavern — Quest Board                            ║
╠══════════════════════════════════════════════════════╣
║  Quest  │ Task                    │ Cat.  │ Pri. │ → ║
║─────────┼─────────────────────────┼───────┼──────┼───║
║  q1     │ REST pros/cons          │ speed │ high │ ⟳ ║
║  q2     │ GraphQL pros/cons       │ speed │ high │ ✓ ║
║  q3     │ Real-world case studies │ research│norm│ ⟳ ║
║  q4     │ Comparison table        │ reason│ crit │ ◇ ║
╠══════════════════════════════════════════════════════╣
║  ⟳ Running: 2  ✓ Done: 1  ◇ Waiting: 1             ║
╚══════════════════════════════════════════════════════╝
```

Status indicators:
- `◇` — waiting (dependencies not met)
- `⟳` — running (dispatched to adventurer)
- `✓` — completed successfully
- `✗` — failed (fallback exhausted)
- `↻` — retrying (falling to next provider)

The DAG is shown automatically when execution begins. The display updates live as quests complete. After all quests finish, the Guild Master's synthesized answer is printed below.

### Setup Wizard

On first run (or `tavern setup`), an interactive rich wizard:
1. Prompts for Anthropic API key (Guild Master) — validates with a test call
2. Prompts for each adventurer's API key — validates each, skip to exclude
3. Auto-detects Ollama if running locally
4. Saves to `~/.tavern/config.yaml`

### Anthropic API Key Resolution

1. Check `ANTHROPIC_API_KEY` environment variable
2. Fall back to `~/.tavern/config.yaml`
3. If neither found, prompt to run `tavern setup`

### Guild Master Model

Default: `claude-sonnet-4-5-20250514` (good balance of speed and quality for decomposition/synthesis).

Override with `--model`:
```bash
tavern --model claude-opus-4-0-20250514 "Complex analysis task"
```

---

## 6. Architecture & Data Flow

### Two Claude API Calls

```
User prompt + stdin
       │
       ▼
┌─────────────────────┐
│  Courtyard          │  (prompt + piped input)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Guild Master       │  Claude API call #1: Decompose
│  (Anthropic API)    │  → Analyzes Courtyard
│                     │  → Produces quest DAG with curated context
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Bulletin Board     │  DAG displayed in terminal (rich)
│  (Quest DAG)        │  User sees it auto-execute
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Executor           │  Wave-based parallel execution
│  (ThreadPoolExecutor)│  Priority ordering within waves
│                     │  Fallback on failure
│  ┌───┐ ┌───┐ ┌───┐ │
│  │ q1│ │ q2│ │ q3│ │  Wave 1 (independent quests)
│  └─┬─┘ └─┬─┘ └─┬─┘ │
│    │     │     │    │
│    ▼     ▼     ▼    │
│  Groq  Mistral Gemini│  Each gets only its curated context
│    │     │     │    │
│    └──┬──┘─────┘    │
│       ▼             │
│    ┌────┐           │
│    │ q4 │           │  Wave 2 (depends on q1-q3)
│    └──┬─┘           │
│       │             │
│    OpenRouter       │
└───────┬─────────────┘
        │
        ▼
┌─────────────────────┐
│  Guild Master       │  Claude API call #2: Synthesize
│  (Anthropic API)    │  → Reviews all quest outputs
│                     │  → Produces unified answer
└─────────┬───────────┘
        │
        ▼
   Final answer printed to terminal
```

### Provider Communication

All provider calls use **httpx** (synchronous, within ThreadPoolExecutor threads):
- POST to provider's chat completions endpoint
- API key in Authorization header (or query param for Gemini)
- 30-second timeout per call
- On failure: log error, try next provider in ranking chain

### Quality Assurance

During synthesis (API call #2), the Guild Master reviews all quest outputs. If a result is low quality:
- The Guild Master notes this in the final answer
- Future enhancement: automatic retry with `skip_providers`

If a quest has `fallback_exhausted: true` (all providers failed), the Guild Master handles it directly in the synthesis — it has the full Courtyard and can answer any quest itself.

---

## 7. Configuration

### Config File: `~/.tavern/config.yaml`

```yaml
guild_master:
  api_key: "sk-ant-..."        # or use ANTHROPIC_API_KEY env var
  model: "claude-sonnet-4-5-20250514"

adventurers:
  groq:
    api_key: "gsk_..."
  cerebras:
    api_key: "csk-..."
  sambanova:
    api_key: "..."
  gemini:
    api_key: "..."
  openrouter:
    api_key: "sk-or-..."
  mistral:
    api_key: "..."
  github_models:
    api_key: "ghp_..."         # GitHub personal access token
  zhipu:
    api_key: "..."
  ollama:
    enabled: true              # auto-detected, no key needed
```

Only configured adventurers participate in routing. Skipped providers are excluded from the ranking chain.

---

## 8. Project Structure

```
~/tavern/
├── pyproject.toml              # hatchling build, dependencies
├── README.md
├── .gitignore
├── src/tavern/
│   ├── __init__.py             # __version__
│   ├── __main__.py             # python -m tavern entry point
│   ├── cli.py                  # Arg parsing, rich live display, main flow
│   ├── courtyard.py            # Courtyard dataclass (prompt + stdin)
│   ├── guildmaster.py          # Anthropic API: decompose() + synthesize()
│   ├── dag.py                  # Quest schema (Pydantic), validate, compute_waves
│   ├── executor.py             # Priority-aware wave executor (ThreadPoolExecutor)
│   ├── fallback.py             # CATEGORY_RANKINGS, get_provider_chain(), get_model_for_provider()
│   ├── config.py               # TavernConfig, load_config(), save_config()
│   ├── wizard.py               # Interactive setup wizard (rich)
│   └── providers/
│       ├── __init__.py         # PROVIDER_REGISTRY
│       ├── base.py             # BaseProvider ABC, ProviderResult
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
    ├── test_guildmaster.py
    ├── test_dag.py
    ├── test_executor.py
    ├── test_fallback.py
    ├── test_config.py
    ├── test_wizard.py
    ├── test_cli.py
    └── test_providers.py
```

### Dependencies

- **httpx** — all provider HTTP calls
- **rich** — CLI output, live display, setup wizard
- **pydantic** — quest DAG schema validation
- **pyyaml** — config file
- **anthropic** — Guild Master API calls (official SDK)

Dev dependencies:
- **pytest** + **pytest-asyncio**
- **respx** — HTTP mocking for provider tests

### Reuse from llmx

The following patterns carry over from the existing `~/llmx/` codebase:

| Component | Reuse | Changes |
|-----------|-------|---------|
| DAG schema (`dag.py`) | Structure | Add `context`, `priority` fields; rename node→quest, `nX`→`qX` |
| Executor (`executor.py`) | Pattern | Add priority ordering within waves |
| Provider base (`providers/base.py`) | Direct | Same ABC + ProviderResult |
| Fallback routing (`fallback.py`) | Structure | Updated rankings for 9 providers |
| Config (`config.py`) | Pattern | New path (`~/.tavern/`), add `guild_master` section |
| Wizard (`wizard.py`) | Pattern | Add Anthropic key step, 9 providers |
| Groq provider | Direct | Minor updates |
| Gemini provider | Direct | Minor updates |
| OpenRouter provider | Direct | Minor updates |
| Ollama provider | Direct | Minor updates |

New components (no llmx equivalent):
- `courtyard.py` — new
- `guildmaster.py` — new (Anthropic API integration)
- `cli.py` — new (replaces MCP server as the interface)
- `cerebras.py`, `sambanova.py`, `mistral.py`, `github_models.py`, `zhipu.py` — new providers

---

## 9. Future Enhancements (Out of Scope for v1)

These are explicitly **not** part of the initial implementation:

- Claude Code MCP plugin version (wrap tavern as an MCP server)
- Streaming output from adventurers
- Automatic retry on low-quality results during synthesis
- Quest result caching
- Custom provider ranking overrides per user
- Web UI / TUI dashboard
