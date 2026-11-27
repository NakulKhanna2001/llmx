"""Exercise the installed protocol and real HTTP adapter with a local model stand-in."""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class LocalModel(BaseHTTPRequestHandler):
    def reply(self, body, status=200):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/api/tags":
            self.reply({"models": [{"name": "fixture:local"}]})
        else:
            self.reply({}, 404)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            self.reply({"capabilities": ["completion"]})
        elif self.path == "/api/generate" and data.get("model") == "fixture:local":
            self.reply({"response": f"answer({data['prompt']})", "done": True})
        else:
            self.reply({}, 404)

    def log_message(self, *args):
        pass


@pytest.mark.asyncio
async def test_fresh_stdio_session_runs_a_dag_and_regenerates_its_dependent(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    with ThreadingHTTPServer(("127.0.0.1", 0), LocalModel) as model:
        thread = Thread(target=model.serve_forever, daemon=True)
        thread.start()
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "llmx", "serve"], cwd=tmp_path,
            env={"PYTHONPATH": str(source), "HOME": str(tmp_path), "USERPROFILE": str(tmp_path),
                 "OLLAMA_HOST": f"http://127.0.0.1:{model.server_port}"},
        )
        try:
            async with stdio_client(params) as (reader, writer), ClientSession(reader, writer) as client:
                async def call(name, arguments=None):
                    result = await client.call_tool(name, arguments or {})
                    assert not result.isError
                    return result.content[0].text

                await client.initialize()
                tools = await client.list_tools()
                assert {t.name for t in tools.tools} == {
                    "llmx_setup", "llmx_status", "llmx_preview_dag", "llmx_execute_dag", "llmx_retry_node",
                }
                assert json.loads(await call("llmx_setup"))["active_providers"] == ["ollama"]
                assert json.loads(await call("llmx_status"))["configured"] == ["ollama"]
                dag = {"id": "demo", "prompt": "Summarize the findings", "nodes": [
                    {"id": "summary", "task": "Summarize {facts}", "category": "reasoning", "depends_on": ["facts"]},
                    {"id": "facts", "task": "facts", "category": "speed"},
                ]}
                preview = await call("llmx_preview_dag", {"dag": dag})
                assert "2 waves" in preview
                execution = json.loads(await call("llmx_execute_dag", {"dag": dag}))
                assert execution["results"]["summary"]["output"] == "answer(Summarize answer(facts))"
                run_id = execution["execution_id"]
                corrected = json.loads(await call("llmx_retry_node", {"execution_id": run_id,
                                             "node_id": "facts", "replacement_output": "checked fact"}))
                assert corrected["invalidated_nodes"] == ["summary"]
                rerun = json.loads(await call("llmx_retry_node", {"execution_id": run_id, "node_id": "summary"}))
                assert rerun["output"] == "answer(Summarize checked fact)"
        finally:
            model.shutdown()
            thread.join()
