import httpx
import pytest
import respx

from llmx.providers.base import ProviderResult
from llmx.providers.groq import GroqProvider
from llmx.providers.openai import OpenAIProvider


@respx.mock
def test_groq_complete_success():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "Hello from Groq"}}],
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


@respx.mock
def test_groq_validate_success():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 1},
        })
    )
    provider = GroqProvider(api_key="gsk_test")
    assert provider.validate() is True


@respx.mock
def test_groq_validate_bad_key():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "invalid key"}})
    )
    provider = GroqProvider(api_key="bad_key")
    assert provider.validate() is False


@respx.mock
def test_openai_complete_success():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "Hello from OpenAI"}}],
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


from llmx.providers.gemini import GeminiProvider


@respx.mock
def test_gemini_complete_success():
    respx.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent").mock(
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
    respx.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    provider = GeminiProvider(api_key="gem_test")
    result = provider.complete("Say hello")
    assert result.success is False
    assert result.error_code == 429


from llmx.providers.openrouter import OpenRouterProvider


@respx.mock
def test_openrouter_complete_success():
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "Hello from OpenRouter"}}],
            "usage": {"total_tokens": 10},
        })
    )
    provider = OpenRouterProvider(api_key="sk-or-test")
    result = provider.complete("Say hello", model="deepseek/deepseek-coder")
    assert result.output == "Hello from OpenRouter"
    assert result.success is True


@respx.mock
def test_openrouter_validate_success():
    respx.get("https://openrouter.ai/api/v1/auth/key").mock(
        return_value=httpx.Response(200, json={"data": {"label": "test"}})
    )
    provider = OpenRouterProvider(api_key="sk-or-test")
    assert provider.validate() is True


@respx.mock
def test_openrouter_validate_bad_key():
    respx.get("https://openrouter.ai/api/v1/auth/key").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    provider = OpenRouterProvider(api_key="bad")
    assert provider.validate() is False
