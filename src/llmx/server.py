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
