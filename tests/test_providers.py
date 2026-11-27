import json

import httpx
import pytest
import respx

from llmx.providers.groq import GroqProvider
from llmx.providers.openai import OpenAIProvider
from llmx.providers.gemini import GeminiProvider
from llmx.providers.ollama import OllamaProvider
from llmx.providers.openrouter import OpenRouterProvider


@respx.mock
def test_groq_complete_success():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": "Hello from Groq"}}],
            "usage": {"total_tokens": 10},
        })
    )
    provider = GroqProvider(api_key="gsk_test")
    result = provider.complete("Say hello")
    assert result.output == "Hello from Groq"
    assert result.success is True


@respx.mock
def test_groq_complete_rate_limit():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    provider = GroqProvider(api_key="gsk_test")
    result = provider.complete("Say hello")
    assert result.success is False
    assert result.error_code == 429


@pytest.mark.parametrize(("provider", "models_url"), [
    (GroqProvider("test-key"), "https://api.groq.com/openai/v1/models"),
    (OpenAIProvider("test-key"), "https://api.openai.com/v1/models"),
])
@respx.mock
def test_key_validation_uses_catalog_without_generation(provider, models_url):
    route = respx.get(models_url).respond(200, json={"data": [{"id": provider.default_model}]})
    assert provider.validate()
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-key"
    route.respond(401)
    assert not provider.validate()


@respx.mock
def test_openai_complete_success():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": "Hello from OpenAI"}}],
            "usage": {"total_tokens": 10},
        })
    )
    provider = OpenAIProvider(api_key="sk-test")
    result = provider.complete("Say hello")
    assert result.output == "Hello from OpenAI"
    assert result.success is True


@respx.mock
def test_openai_complete_rate_limit():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    provider = OpenAIProvider(api_key="sk-test")
    result = provider.complete("Say hello")
    assert result.success is False
    assert result.error_code == 429


@respx.mock
def test_gemini_complete_success():
    respx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{GeminiProvider.default_model}:generateContent").mock(
        return_value=httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "Hello from Gemini"}]}}],
        })
    )
    provider = GeminiProvider(api_key="gem_test")
    result = provider.complete("Say hello")
    assert result.output == "Hello from Gemini"
    assert result.success is True


@respx.mock
def test_gemini_complete_rate_limit():
    respx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{GeminiProvider.default_model}:generateContent").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    provider = GeminiProvider(api_key="gem_test")
    result = provider.complete("Say hello")
    assert result.success is False
    assert result.error_code == 429


@respx.mock
def test_openrouter_complete_success():
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": "Hello from OpenRouter"}}],
            "usage": {"total_tokens": 10},
        })
    )
    provider = OpenRouterProvider(api_key="sk-or-test")
    result = provider.complete("Say hello", model="alternative/model")
    assert result.output == "Hello from OpenRouter"
    assert result.success is True


@respx.mock
def test_openrouter_validate_success():
    respx.get("https://openrouter.ai/api/v1/key").mock(
        return_value=httpx.Response(200, json={"data": {"label": "test"}})
    )
    provider = OpenRouterProvider(api_key="sk-or-test")
    assert provider.validate() is True


@respx.mock
def test_openrouter_validate_bad_key():
    respx.get("https://openrouter.ai/api/v1/key").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    provider = OpenRouterProvider(api_key="bad")
    assert provider.validate() is False


@respx.mock
def test_ollama_complete_success():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json={
            "response": "Hello from Ollama",
            "done": True,
        })
    )
    provider = OllamaProvider(model="llama3")
    result = provider.complete("Say hello")
    assert result.output == "Hello from Ollama"
    assert result.success is True


@respx.mock
def test_ollama_validate_running():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={
            "models": [{"name": "llama3"}, {"name": "codellama"}],
        })
    )
    respx.post("http://localhost:11434/api/show").respond(200, json={"capabilities": ["completion"]})
    provider = OllamaProvider()
    is_valid, models = provider.validate_with_models()
    assert is_valid is True
    assert "llama3" in models


@respx.mock
def test_ollama_validate_not_running():
    respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    provider = OllamaProvider()
    is_valid, models = provider.validate_with_models()
    assert is_valid is False
    assert models == []


COMPATIBLE_APIS = [
    (GroqProvider("secret-token"), "https://api.groq.com/openai/v1/chat/completions"),
    (OpenAIProvider("secret-token"), "https://api.openai.com/v1/chat/completions"),
    (OpenRouterProvider("secret-token"), "https://openrouter.ai/api/v1/chat/completions"),
]


@pytest.mark.parametrize(("provider", "url"), COMPATIBLE_APIS)
@respx.mock
def test_chat_api_failures_are_redacted_and_allow_fallback(provider, url):
    route = respx.post(url)
    for response in [
        httpx.Response(403, text="invalid token secret-token"),
        httpx.Response(200, text="secret-token is not JSON"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}),
        httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "  "}}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": "partial"}}]}),
    ]:
        route.mock(return_value=response)
        result = provider.complete("prompt", "custom-model")
        assert not result.success and result.output == "" and result.error_message
        assert "secret-token" not in result.error_message
        assert result.error_code == (403 if response.status_code == 403 else None)
        assert result.model == "custom-model"
        assert json.loads(route.calls.last.request.content)["model"] == "custom-model"

    route.mock(side_effect=httpx.ConnectError("endpoint secret-token"))
    result = provider.complete("prompt")
    assert not result.success and "secret-token" not in result.error_message


@respx.mock
def test_ollama_lists_only_installed_local_completion_models(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:33331/")
    root = "http://127.0.0.1:33331"
    tags = respx.get(root + "/api/tags")
    tags.respond(200, json={"models": [None, {}, {"name": " "}, {"name": "embed"},
                                       {"name": "remote"}, {"name": "text"}, {"name": "text"}]})
    respx.post(root + "/api/show", json={"model": "embed"}).respond(200, json={"capabilities": ["embedding"]})
    respx.post(root + "/api/show", json={"model": "remote"}).respond(
        200, json={"capabilities": ["completion"], "remote_model": "hosted"}
    )
    respx.post(root + "/api/show", json={"model": "text"}).respond(200, json={"capabilities": ["completion"]})
    provider = OllamaProvider(model="text")
    assert provider.validate_with_models() == (True, ["text"])
    tags.respond(200, json={"models": []})
    assert not provider.validate()
    tags.respond(200, text="bad json")
    assert not provider.validate()
    generate = respx.post(root + "/api/generate")
    generate.respond(200, json={"response": "partial", "done": False})
    assert not provider.complete("test").success
    generate.respond(200, json={"response": "partial", "done": True, "done_reason": "length"})
    assert not provider.complete("test").success
    generate.respond(200, json={"response": " ", "done": True})
    assert not provider.complete("test").success
    generate.respond(200, json={"response": "done", "done": True, "done_reason": "stop"})
    assert provider.complete("test").output == "done"
    generate.respond(500, text="sensitive-provider-data")
    error = provider.complete("test")
    assert not error.success and "sensitive-provider-data" not in error.error_message


@respx.mock
def test_ollama_uses_capabilities_and_cloud_metadata_in_tags(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    respx.get("http://localhost:11434/api/tags").respond(200, json={"models": [
        {"name": "local", "capabilities": ["completion"]},
        {"name": "embed", "capabilities": ["embedding"]},
        {"name": "cloud", "remote_model": "cloud", "capabilities": ["completion"]},
        {"name": "remote-alias", "remote_host": "https://ollama.com"},
    ]})
    assert OllamaProvider().validate_with_models() == (True, ["local"])
