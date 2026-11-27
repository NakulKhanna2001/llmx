import asyncio
import json
import shlex
import sys
from dataclasses import asdict, dataclass, field
from threading import Lock
from uuid import uuid4

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .config import DEFAULT_CONFIG_PATH, LlmxConfig, load_config
from .dag import Dag, DagValidationError, format_dag_preview, validate_dag
from .executor import DagExecutor, ExecutionResult
from .providers import PROVIDER_REGISTRY
from .providers.base import BaseProvider
from .wizard import discover_ollama

server = Server("llmx")


@dataclass
class SavedRun:
    executor: DagExecutor
    result: ExecutionResult
    # A cancelled MCP request can leave its worker running; serialize in the worker.
    lock: Lock = field(default_factory=Lock)

    def retry(self, node_id, skip_providers, replacement_output):
        with self.lock:
            return self.executor.retry_node(self.result, node_id, skip_providers, replacement_output)


# stdio serves one client. IDs keep overlapping calls from sharing node outputs.
# Bound the lifetime of provider objects, prompt data and attachments kept for retries.
RECENT_RUN_LIMIT = 8
_runs: dict[str, SavedRun] = {}


def _runtime_config() -> LlmxConfig:
    config = load_config()
    config.providers["ollama"] = discover_ollama()
    return config


def _build_providers() -> tuple[dict[str, BaseProvider], list[str]]:
    config = _runtime_config()
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

    return providers, list(providers)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="llmx_setup",
            description="Detect local Ollama models and show how to configure optional provider keys privately in a terminal",
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
            description="Retry a node or record a client-provided answer. Returns affected downstream nodes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "execution_id": {
                        "type": "string",
                        "description": "ID returned by execute. Defaults to the last completed run.",
                    },
                    "replacement_output": {
                        "type": "string", "minLength": 1,
                        "description": "Supply a corrected node answer directly without a provider call.",
                    },
                    "skip_providers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Override the default list of providers already tried for this input.",
                    },
                },
                "required": ["node_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    if name in {"llmx_setup", "llmx_status"}:
        config = await asyncio.to_thread(_runtime_config)
        available = [p for p in config.get_available_providers() if p in PROVIDER_REGISTRY]
        status = {
            "configured": available,
            "missing": [p for p in PROVIDER_REGISTRY if p not in available],
            "config_path": str(DEFAULT_CONFIG_PATH),
        }
        if name == "llmx_setup":
            status["active_providers"] = available
            argv = [sys.executable, "-m", "llmx", "setup"]
            status["setup_argv"] = argv
            if sys.platform == "win32":
                executable = sys.executable.replace("'", "''")
                status["setup_command"] = f"& '{executable}' -m llmx setup"
                terminal = "PowerShell"
            else:
                status["setup_command"] = shlex.join(argv)
                terminal = "your terminal"
            status["instructions"] = (
                f"Run the setup command in {terminal} to enter optional cloud keys. "
                "Ollama models installed locally are detected automatically."
            )
        return [TextContent(type="text", text=json.dumps(status, indent=2))]

    if name == "llmx_preview_dag":
        try:
            preview = format_dag_preview(Dag(**arguments["dag"]))
            return [TextContent(type="text", text=preview)]
        except (ValueError, TypeError, KeyError, DagValidationError) as exc:
            return [TextContent(type="text", text=f"DAG validation error: {exc}")]

    if name == "llmx_execute_dag":
        try:
            dag = Dag(**arguments["dag"])
            validate_dag(dag)
            providers, available = await asyncio.to_thread(_build_providers)
            executor = DagExecutor(dag, providers, available)
            result = await asyncio.to_thread(executor.execute)
            execution_id = uuid4().hex
            _runs[execution_id] = SavedRun(executor, result)
            if len(_runs) > RECENT_RUN_LIMIT:
                del _runs[next(iter(_runs))]
            output = {
                "execution_id": execution_id,
                "results": {node_id: asdict(value) for node_id, value in result.results.items()},
                "execution_time_ms": result.execution_time_ms,
                "retries": result.retries,
            }
            return [TextContent(type="text", text=json.dumps(output, indent=2))]
        except Exception as exc:
            return [TextContent(type="text", text=f"Execution error: {exc}")]

    if name == "llmx_retry_node":
        execution_id = arguments.get("execution_id")
        if execution_id is None:
            execution_id = next(reversed(_runs), None)
        if execution_id is None:
            return [TextContent(type="text", text="No previous DAG execution found")]
        run = _runs.get(execution_id)
        if run is None:
            return [TextContent(type="text", text="Unknown or expired execution_id")]
        node_id = arguments["node_id"]
        if node_id not in run.executor.nodes:
            return [TextContent(type="text", text=f"Node {node_id} not found in DAG")]
        replacement = arguments.get("replacement_output")
        if replacement is not None and (not isinstance(replacement, str) or not replacement.strip()):
            return [TextContent(type="text", text="replacement_output must be nonblank text")]
        value, invalidated = await asyncio.to_thread(
            run.retry, node_id, arguments.get("skip_providers"), replacement
        )
        output = {
            "execution_id": execution_id, "node_id": node_id,
            **asdict(value), "invalidated_nodes": invalidated,
        }
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
