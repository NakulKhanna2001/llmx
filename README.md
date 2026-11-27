# llmx

llmx is a Claude Code plugin for questions that benefit from several models working together. Claude breaks a prompt into tasks, shows you the plan, and synthesizes the results. A Python MCP server runs independent tasks in parallel and passes earlier answers to the tasks that depend on them.

## Install and run

You need Python 3.11 or newer and Claude Code. This macOS/Linux example uses `python3.13`. Substitute your installed Python 3.11+ executable if needed. On Windows, activate the virtual environment with `.\.venv\Scripts\Activate.ps1` in PowerShell and pass the checkout’s absolute path to `--plugin-dir`.

```bash
git clone https://github.com/NakulKhanna2001/llmx.git
cd llmx
LLMX_ROOT="$PWD"
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m llmx setup
claude --plugin-dir "$LLMX_ROOT"
```

Enter cloud keys at the hidden prompts in your own terminal. The wizard checks the keys you enter with their providers and saves the configuration locally at `~/.llmx/config.yaml`. Treat this file as a credential file. You can skip every cloud provider and use a model already installed in a running Ollama instance on `localhost:11434`. Cloud inference can incur provider charges. The default Ollama connection stays on your machine. If you set `OLLAMA_HOST`, requests go to that server.

The plugin registers the installed `llmx serve` executable automatically. Start Claude from the activated environment so the executable is on `PATH`. You can work in any directory. For example, in a new terminal with an existing checkout:

```bash
LLMX_ROOT="/absolute/path/to/llmx"
source "$LLMX_ROOT/.venv/bin/activate"
claude --plugin-dir "$LLMX_ROOT"
```

In Claude Code, `/mcp` should list `plugin:llmx:llmx`. Run the skill with a request:

```text
/llmx Compare REST and GraphQL for our mobile app. Explain when each is a better fit.
```

The full command is `/llmx:llmx`. Claude Code also accepts the short `/llmx` when that name is free. Loading the plugin supplies both the slash command and the MCP tools. [Claude Code skill names](https://code.claude.com/docs/en/skills#how-a-skill-gets-its-command-name) · [Plugin loading](https://code.claude.com/docs/en/plugins/create#load-a-plugin-for-one-session)

For another MCP client, configure a **stdio** server with the installed `llmx` executable and the single argument `serve`. Give that client the absolute path to `.venv/bin/llmx` (or `.venv\Scripts\llmx.exe` on Windows) if it launches outside your activated environment.

## What happens to a request

For the example above, Claude could propose this plan before any provider executes it:

| Node | Task | Category | Depends on |
| --- | --- | --- | --- |
| `n1` | Explain REST trade-offs for this app | `speed` | — |
| `n2` | Explain GraphQL trade-offs for this app | `speed` | — |
| `n3` | Find supporting examples | `research` | — |
| `n4` | Compare the findings in `{n1}`, `{n2}` and `{n3}` | `reasoning` | `n1`, `n2`, `n3` |

After you approve, the first three nodes run together. The fourth receives their actual outputs. For a Gemini task, a caller can also supply inline PNG, JPEG, WebP or PDF content in an optional `attachments` array. Each attachment contains `mime_type` and standard base64 `data`. A node accepts up to four attachments, each up to 3 MiB before encoding. A filename or URL in a prompt is just text.

### Routing and recovery

| Category | First choice when available |
| --- | --- |
| `speed` | Groq |
| `code` | DeepSeek via OpenRouter |
| `research` | Perplexity Sonar via OpenRouter |
| `large_context` | Gemini |
| `reasoning`, `creative` | OpenAI |

The server ranks compatible, configured providers for each node. It moves to the next provider on failure. Ollama can handle text tasks on the configured host. A model’s response to a research task still needs its sources checked.

Claude reviews the answers after execution. For a weak result, `llmx_retry_node` tries another provider using the latest dependency outputs. A successful correction identifies affected downstream nodes so Claude can refresh them in dependency order. If every eligible provider fails, Claude can solve that node in-session and explicitly record the answer with `replacement_output`. Use the returned `execution_id` to keep retries attached to their original run. The server retains its eight most recently completed runs until it exits.

### MCP tools

| Tool | Purpose |
| --- | --- |
| `llmx_status` | List configured cloud providers and check for installed local Ollama models. |
| `llmx_setup` | Detect local Ollama and report how to start the private terminal wizard. |
| `llmx_preview_dag` | Validate the DAG and show its execution waves for review. |
| `llmx_execute_dag` | Run the approved plan and return the execution ID and per-node results. |
| `llmx_retry_node` | Retry one node or record a Claude replacement and identify stale downstream results. |

## Development

From an activated environment in the checkout:

```bash
python -m pip install -e '.[dev]'
python -m pytest
claude plugin validate .
```

To try setup, preview, execution and a corrected dependency without a cloud API key, run `python -m pytest -q tests/test_stdio_workflow.py`. This test launches the real MCP server and a local HTTP test double.

## License

MIT
