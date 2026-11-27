---
name: llmx
description: Break a request into dependent tasks, preview the plan, run it across configured LLM providers, and synthesize the results in Claude Code.
argument-hint: "[request]"
---

Use the llmx MCP tools for the request below. In the bundled plugin they have the prefix `mcp__plugin_llmx_llmx__`.

1. Call `llmx_status` to see which providers can run tasks. If none are available, call `llmx_setup`. It checks for local Ollama models and returns terminal setup instructions. Have the user enter provider keys privately in their terminal. Keep credentials out of chat, task prompts and MCP arguments. If the user wants to continue with Claude alone, explain that Claude will perform the work in-session.
2. Build a DAG with `id`, `prompt`, and `nodes`. Give each node a unique `id` (for example, `n1`), a `task`, a `category`, and a list of `depends_on` IDs. Include `{n1}` in a task where the first node’s output belongs. Use `speed` for summaries, `code` for programming, `research` for sourcing, `large_context` for long inputs, `reasoning` for analysis, and `creative` for writing. Independent nodes can run together. Pass only inputs needed for the request. For an explicitly supplied image or PDF, the node accepts `attachments` with `mime_type` and base64 `data` for Gemini.
3. Call `llmx_preview_dag` with the `dag`. Show the user the planned work, dependencies and any external inputs the providers will receive. Invite edits or cancellation. Once they confirm, call `llmx_execute_dag` with the agreed DAG and keep its `execution_id` for retries.
4. Review results in dependency order. Check the answer against the task, and check cited evidence when the answer depends on current facts. For an inadequate successful answer, call `llmx_retry_node` with `execution_id` and `node_id`. The retry uses current dependency outputs and skips providers already tried by default. The optional `skip_providers` list overrides those defaults. When `fallback_exhausted` is true, complete the task in the Claude session using its current inputs, then call `llmx_retry_node` with your answer as `replacement_output`. Attribute that result to Claude. If the required evidence or capability is missing, report the gap.
5. Retry any nodes in `invalidated_nodes` in dependency order so downstream reasoning uses the revised upstream result. A node with `blocked_by` waits for those dependencies to succeed. Use the latest accepted outputs to synthesize the answer. Include compact execution details (provider, model, latency, and any retries or Claude replacements) where useful.

User request:

$ARGUMENTS
