import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, local
from types import SimpleNamespace

import pytest

from llmx import server, wizard
from llmx.config import LlmxConfig, load_config, save_config


def simultaneous_wizards(monkeypatch, path, answers_a, answers_b):
    barrier = Barrier(2)
    session = local()

    def synchronized_load(path):
        snapshot = load_config(path)
        barrier.wait(timeout=5)
        return snapshot

    class ValidProvider:
        def __init__(self, api_key):
            pass

        def validate(self):
            return True

    monkeypatch.setattr(wizard, "load_config", synchronized_load)
    monkeypatch.setattr(wizard.Prompt, "ask", lambda *args, **kwargs: next(session.answers))
    monkeypatch.setattr(wizard.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(wizard, "discover_ollama", lambda: {"available": False, "models": []})
    monkeypatch.setattr(wizard, "PROVIDER_REGISTRY", dict.fromkeys(wizard.CLOUD_PROVIDERS, ValidProvider))

    def run(answers):
        session.answers = iter(answers)
        return wizard.run_wizard(path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        return [pool.submit(run, answers) for answers in (answers_a, answers_b)]


def test_separate_wizard_updates_are_merged(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    running = simultaneous_wizards(monkeypatch, path, ["synthetic-a", "", "", ""],
                                   ["", "synthetic-b", "", ""])
    for future in running:
        future.result()
    assert set(load_config(path).get_available_providers()) == {"groq", "openai"}


def test_conflicting_wizards_fail_without_partial_writes_or_secret_echo(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    save_config(LlmxConfig({"groq": {"api_key": "old", "validated": True}}), path)
    running = simultaneous_wizards(monkeypatch, path, ["-", "synthetic-a", "", ""],
                                   ["synthetic-b", "", "synthetic-c", ""])
    errors = [f.exception() for f in running if f.exception() is not None]
    assert len(errors) == 1 and isinstance(errors[0], ValueError)
    assert "groq" in str(errors[0]) and "synthetic" not in str(errors[0])
    providers = set(load_config(path).get_available_providers())
    assert providers in ({"openai"}, {"groq", "gemini"})


@pytest.mark.asyncio
async def test_windows_setup_command_runs_in_powershell(monkeypatch):
    executable = r"C:\Users\Nakul's Files\python.exe"
    monkeypatch.setattr(server, "sys", SimpleNamespace(platform="win32", executable=executable))
    monkeypatch.setattr(server, "_runtime_config", LlmxConfig)
    block, = await server.call_tool("llmx_setup", {})
    result = json.loads(block.text)
    assert result["setup_argv"] == [executable, "-m", "llmx", "setup"]
    assert result["setup_command"] == r"& 'C:\Users\Nakul''s Files\python.exe' -m llmx setup"
    assert "PowerShell" in result["instructions"]
