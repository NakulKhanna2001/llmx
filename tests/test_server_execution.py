import asyncio
import json

import pytest

from llmx import server
from llmx.providers.base import ProviderResult


class Echo:
    def __init__(self, name):
        self.name = name

    def complete(self, prompt, model=None):
        return ProviderResult(output=f"{self.name}: {prompt}", success=True,
                              provider=self.name, model=model or "test")


async def invoke(name, **args):
    content = await server.call_tool(name, args)
    return json.loads(content[0].text)


def graph(prompt):
    return {"id": "shared", "prompt": prompt, "nodes": [
        {"id": "root", "task": prompt, "category": "speed"},
        {"id": "child", "task": "use {root}", "category": "reasoning", "depends_on": ["root"]},
        {"id": "end", "task": "final {child}", "category": "code", "depends_on": ["child"]},
    ]}


@pytest.mark.asyncio
async def test_concurrent_runs_and_quality_retries_keep_their_own_outputs(monkeypatch):
    providers = {name: Echo(name) for name in ["groq", "openai", "openrouter"]}
    monkeypatch.setattr(server, "_build_providers", lambda: (providers, list(providers)))
    first, second = await asyncio.gather(*[
        invoke("llmx_execute_dag", dag=graph(p)) for p in ["first", "second"]
    ])
    a, b = first["execution_id"], second["execution_id"]
    assert a != b
    improved = await invoke("llmx_retry_node", execution_id=a, node_id="root")
    assert improved["output"] == "openai: first"
    assert improved["invalidated_nodes"] == ["child", "end"]
    blocked = await invoke("llmx_retry_node", execution_id=a, node_id="end")
    assert not blocked["success"] and blocked["blocked_by"] == ["child"]
    fresh = await invoke("llmx_retry_node", execution_id=a, node_id="child")
    assert fresh["output"] == "openai: use openai: first"
    other = await invoke("llmx_retry_node", execution_id=b, node_id="child", skip_providers=["openai"])
    assert other["output"] == "groq: use groq: second"
    final = await invoke("llmx_retry_node", execution_id=a, node_id="end")
    assert "use openai: first" in final["output"]


@pytest.mark.asyncio
async def test_client_fallback_recovers_dependents_without_configured_provider(monkeypatch):
    monkeypatch.setattr(server, "_build_providers", lambda: ({}, []))
    run = await invoke("llmx_execute_dag", dag=graph("example"))
    assert run["results"]["root"]["fallback_exhausted"]
    assert run["results"]["child"]["blocked_by"] == ["root"]
    fixed = await invoke("llmx_retry_node", node_id="root", replacement_output="local answer")
    assert fixed["provider"] == "client" and fixed["output"] == "local answer"
    assert fixed["invalidated_nodes"] == ["child", "end"]
    fixed_child = await invoke("llmx_retry_node", node_id="child", replacement_output="next")
    assert fixed_child["success"]


@pytest.mark.asyncio
async def test_simultaneous_quality_retries_advance_eligible_providers(monkeypatch):
    providers = {name: Echo(name) for name in ["groq", "openai", "gemini"]}
    monkeypatch.setattr(server, "_build_providers", lambda: (providers, list(providers)))
    run = await invoke("llmx_execute_dag", dag=graph("query"))
    calls = [invoke("llmx_retry_node", execution_id=run["execution_id"], node_id="root")
             for _ in range(2)]
    results = await asyncio.gather(*calls)
    assert {item["provider"] for item in results} == {"openai", "gemini"}
    exhausted = await invoke("llmx_retry_node", execution_id=run["execution_id"], node_id="root")
    assert not exhausted["success"] and exhausted["fallback_exhausted"]
    recovered_child = await invoke("llmx_retry_node", execution_id=run["execution_id"], node_id="child")
    assert recovered_child["output"] == "openai: use gemini: query"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["llmx_preview_dag", "llmx_execute_dag"])
async def test_validation_errors_do_not_echo_attachment_contents(tool):
    malformed = graph("prompt")
    malformed["nodes"][0]["attachments"] = [{"mime_type": "image/png", "data": "sensitive-contents!"}]
    content = await server.call_tool(tool, {"dag": malformed})
    assert "base64" in content[0].text
    assert "sensitive-contents" not in content[0].text


@pytest.mark.asyncio
async def test_attachment_forwarded_on_execution_and_retry(monkeypatch):
    class MediaEcho(Echo):
        def complete_with_attachments(self, prompt, attachments, model=None):
            return self.complete(f"{prompt} [{attachments[0].mime_type}]", model=model)

    providers = {name: MediaEcho(name) for name in ["gemini", "openai"]}
    monkeypatch.setattr(server, "_build_providers", lambda: (providers, list(providers)))
    dag = graph("describe the image")
    dag["nodes"][0].update(category="large_context",
                           attachments=[{"mime_type": "image/png", "data": "YWJj"}])
    run = await invoke("llmx_execute_dag", dag=dag)
    assert run["results"]["root"]["output"] == "gemini: describe the image [image/png]"
    again = await invoke("llmx_retry_node", execution_id=run["execution_id"], node_id="root")
    assert again["output"] == "openai: describe the image [image/png]"
