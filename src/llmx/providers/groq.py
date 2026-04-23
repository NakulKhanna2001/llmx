import time

import httpx

from .base import BaseProvider, ProviderResult

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(BaseProvider):
    name = "groq"
    default_model = "llama-3.1-70b-versatile"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        start = time.monotonic()
        try:
            resp = httpx.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                },
                timeout=30.0,
            )
            latency = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return ProviderResult(
                    output="",
                    success=False,
                    provider=self.name,
                    model=model,
                    latency_ms=latency,
                    error_code=resp.status_code,
                    error_message=resp.text,
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return ProviderResult(
                output=content,
                success=True,
                provider=self.name,
                model=model,
                latency_ms=latency,
            )
        except httpx.TimeoutException:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="",
                success=False,
                provider=self.name,
                model=model,
                latency_ms=latency,
                error_code=None,
                error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        result = self.complete("Hi", model=self.default_model)
        return result.success
