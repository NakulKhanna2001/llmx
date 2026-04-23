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
