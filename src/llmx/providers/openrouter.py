import time

import httpx

from .base import BaseProvider, ProviderResult

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_AUTH_URL = "https://openrouter.ai/api/v1/auth/key"


class OpenRouterProvider(BaseProvider):
    name = "openrouter"
    default_model = "deepseek/deepseek-coder"

    # Category-specific default models
    MODEL_FOR_CATEGORY = {
        "code": "deepseek/deepseek-coder",
        "research": "perplexity/sonar",
    }

    def __init__(self, api_key: str):
        self.api_key = api_key

    def model_for_category(self, category: str) -> str:
        return self.MODEL_FOR_CATEGORY.get(category, self.default_model)

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        start = time.monotonic()
        try:
            resp = httpx.post(
                OPENROUTER_API_URL,
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
                error_message="Timeout after 30s",
            )

    def validate(self) -> bool:
        try:
            resp = httpx.get(
                OPENROUTER_AUTH_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
