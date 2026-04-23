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
