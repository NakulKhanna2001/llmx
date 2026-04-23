import time

import httpx

from .base import BaseProvider, ProviderResult

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(BaseProvider):
    name = "gemini"
    default_model = "gemini-2.0-flash"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        url = f"{GEMINI_API_BASE}/{model}:generateContent"
        start = time.monotonic()
        try:
            resp = httpx.post(
                url,
                params={"key": self.api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
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
            content = data["candidates"][0]["content"]["parts"][0]["text"]
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
                error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        result = self.complete("Hi", model=self.default_model)
        return result.success
