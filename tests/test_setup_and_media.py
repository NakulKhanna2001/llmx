import json
import os
from stat import S_IMODE
from unittest.mock import patch

import httpx
import pytest
import respx
from pydantic import ValidationError

from llmx import server, wizard
from llmx.attachments import Attachment
from llmx.config import LlmxConfig, load_config, save_config
from llmx.providers.gemini import GEMINI_API_BASE, GeminiProvider
from llmx.providers.openai import OpenAIProvider


def test_wizard_keeps_existing_key_and_saves_privately(tmp_path):
    path = tmp_path / "config.yaml"
    save_config(LlmxConfig({"groq": {"api_key": "old-key", "validated": True}}), path)
    with patch.object(wizard.Prompt, "ask", side_effect=["", "new-key", "", ""]) as prompt, \
            patch.object(OpenAIProvider, "validate", return_value=False), \
            patch.object(wizard, "discover_ollama", return_value={"available": False, "models": []}):
        summary = wizard.run_wizard(path)
    assert summary["active_providers"] == ["groq"]
    assert all(call.kwargs["password"] for call in prompt.call_args_list)
    assert "new-key" not in path.read_text()
    assert load_config(path).providers["groq"]["api_key"] == "old-key"
    if os.name == "posix":
        assert S_IMODE(path.stat().st_mode) == 0o600


def test_malformed_config_never_echoes_credentials_or_overwrites_original(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("providers: [very-private-marker, [")
    with pytest.raises(ValueError) as caught:
        wizard.run_wizard(path)
    assert "very-private-marker" not in str(caught.value)
    assert "very-private-marker" in path.read_text()


@pytest.mark.asyncio
async def test_mcp_setup_does_not_prompt_or_disclose_keys(monkeypatch):
    monkeypatch.setattr(server, "load_config", lambda: LlmxConfig({"openai": {"validated": True, "api_key": "private-marker"}}))
    monkeypatch.setattr(server, "discover_ollama", lambda: {"available": True, "models": ["local"]})
    for name in ("llmx_setup", "llmx_status"):
        block, = await server.call_tool(name, {})
        assert "private-marker" not in block.text
        result = json.loads(block.text)
        assert result["configured"] == ["openai", "ollama"]
        if name == "llmx_setup":
            assert result["setup_command"].endswith("-m llmx setup")


@respx.mock
def test_gemini_transmits_inline_attachment_and_returns_only_answer_parts():
    gemini = GeminiProvider(api_key="private-test-key")
    route = respx.post(f"{GEMINI_API_BASE}/{gemini.default_model}:generateContent").respond(
        json={"candidates": [{"content": {"parts": [{"text": "internal", "thought": True}, {"text": "one"}, {"text": " two"}]}}]})
    media = Attachment(mime_type="image/png", data="AQID")
    result = gemini.complete_with_attachments("Summarize the image", [media])
    assert result.success and result.output == "one two"
    request = route.calls.last.request
    assert request.headers["x-goog-api-key"] == "private-test-key"
    assert "private-test-key" not in str(request.url)
    assert json.loads(request.content)["contents"][0]["parts"] == [
        {"inlineData": {"mimeType": "image/png", "data": "AQID"}}, {"text": "Summarize the image"}]
    assert not OpenAIProvider(api_key="unused").complete_with_attachments("prompt", [media]).success


@respx.mock
def test_gemini_empty_or_invalid_provider_output_is_recoverable():
    gemini = GeminiProvider(api_key="test")
    endpoint = respx.post(f"{GEMINI_API_BASE}/{gemini.default_model}:generateContent")
    for data in ({"candidates": []}, {"error": "blocked"}, {"candidates": [{"content": {"parts": [{}]}}]},
                 {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "partial"}]}}]}):
        endpoint.respond(json=data)
        assert not gemini.complete("test").success
    endpoint.mock(side_effect=httpx.ConnectError("offline"))
    assert not gemini.complete("test").success


def test_attachments_refuse_urls_and_malformed_base64():
    for data in ("https://example.com/a.png", "not-base64"):
        with pytest.raises(ValidationError):
            Attachment(mime_type="image/png", data=data)
