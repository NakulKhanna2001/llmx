"""Adapters for chat-completion APIs with OpenAI-compatible text responses."""

import time
from collections.abc import Callable

import httpx

from .base import BaseProvider, ProviderResult


def request_text(
    provider: str, model: str, url: str, payload: dict, read_text: Callable[[dict], str],
    *, timeout: float, headers: dict[str, str] | None = None,
) -> ProviderResult:
    start = time.monotonic()
    result = ProviderResult(output="", success=False, provider=provider, model=model)
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get("error"):
            raise ValueError("Invalid response")
        output = read_text(data)
        if not isinstance(output, str) or not output.strip():
            raise ValueError("Empty answer")
        result.output = output
        result.success = True
    except httpx.HTTPStatusError as error:
        result.error_code = error.response.status_code
        result.error_message = f"{provider} returned HTTP {result.error_code}."
    except httpx.TimeoutException:
        result.error_message = f"{provider} timed out."
    except (httpx.RequestError, httpx.InvalidURL):
        result.error_message = f"Could not reach {provider}."
    except (ValueError, TypeError, KeyError, IndexError):
        result.error_message = f"{provider} returned an incomplete or invalid answer."
    result.latency_ms = int((time.monotonic() - start) * 1000)
    return result


def chat_text(data: dict) -> str:
    choice = data["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("Generation did not finish normally")
    return choice["message"]["content"]


class ChatProvider(BaseProvider):
    api_url: str
    models_url: str

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        return request_text(
            self.name, model, self.api_url,
            {"model": model, "messages": [{"role": "user", "content": prompt}],
             "max_completion_tokens": 4096},
            chat_text, headers=self.headers, timeout=30.0,
        )

    def validate(self) -> bool:
        try:
            response = httpx.get(self.models_url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            models = response.json()["data"]
            return isinstance(models, list) and any(
                isinstance(model, dict) and model.get("id") == self.default_model for model in models
            )
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            return False
