# llmx

A Claude Code plugin that orchestrates multiple LLM providers in parallel. Submit a complex prompt, and Claude decomposes it into a DAG of subtasks — each routed to the best provider for the job — executes them simultaneously, and synthesizes a unified response.

## How It Works

```
/llmx "Compare REST vs GraphQL for a mobile app backend"

  Claude decomposes into 4 subtasks:
  ┌──────┬────────────────────────────────┬────────────┬──────────┐
  │ Node │ Task                           │ Category   │ Provider │
  ├──────┼────────────────────────────────┼────────────┼──────────┤
  │ n1   │ REST pros/cons for mobile      │ speed      │ Groq     │
  │ n2   │ GraphQL pros/cons for mobile   │ speed      │ Groq     │
  │ n3   │ Real-world case studies         │ research   │ Perplexity│
  │ n4   │ Comparison table (needs n1-n3) │ reasoning  │ OpenAI   │
  └──────┴────────────────────────────────┴────────────┴──────────┘
  n1, n2, n3 run in parallel → n4 runs after → Claude synthesizes
```

## Providers

| Provider | Strength | Key Required |
|----------|----------|-------------|
| Groq | Speed — fast inference for simple tasks | Yes |
| OpenAI | Reasoning — analysis, comparison, creative | Yes |
| Gemini | Large context — long documents, multimodal | Yes |
| OpenRouter | Meta-provider — DeepSeek (code), Perplexity (research) | Yes |
| Ollama | Offline fallback — local models, no API key | No |

Claude (already in your session) handles orchestration, quality checks, and last-resort execution.

## Task Routing

Each subtask is categorized, and providers are ranked per category:

| Category | Best For | Top Pick |
|----------|----------|----------|
| `speed` | Quick lookups, simple tasks | Groq |
| `code` | Code generation, debugging | DeepSeek via OpenRouter |
| `research` | Web-grounded, current info | Perplexity via OpenRouter |
| `large_context` | Long docs, multimodal | Gemini |
| `reasoning` | Analysis, comparison | OpenAI |
| `creative` | Writing, brainstorming | OpenAI |

If the top provider is down or rate-limited, execution automatically falls to the next in ranking.

## Setup

### 1. Install dependencies

```bash
cd ~/llmx
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Register as a Claude Code MCP server

Add this to your `~/.claude/settings.json` under `"mcpServers"`:

```json
{
  "mcpServers": {
    "llmx": {
      "command": "/Users/YOUR_USERNAME/llmx/.venv/bin/python",
      "args": ["-m", "llmx.server"]
    }
  }
}
```

Or add a project-level `.mcp.json` in any repo where you want `/llmx` available.

### 3. Configure providers

On first use, Claude will call `llmx_setup` to run the interactive wizard. It prompts for each API key, validates it with a live test call, and saves to `~/.llmx/config.yaml`. Skipped providers are excluded from routing. Ollama is detected automatically.

## Quality Assurance

Claude reviews all subtask outputs at synthesis time. If a result is low quality, Claude triggers a retry with the next provider in the ranking — no manual intervention needed. If every provider fails, Claude executes the subtask directly.

## Architecture

```
┌─────────────────────────────┐
│    Claude Code Session      │
│  /llmx "prompt"             │
│  Claude decomposes → DAG    │
│  Claude synthesizes results │
└──────────┬──────────────────┘
           │ MCP tools
           ▼
┌──────────────────────────────┐
│    llmx MCP Server (Python)  │
│  Setup wizard                │
│  DAG validation + preview    │
│  Parallel execution          │
│  Provider fallback           │
└──────────────────────────────┘
```

## License

MIT
