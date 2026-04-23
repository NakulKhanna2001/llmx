import time

import httpx

from .base import BaseProvider, ProviderResult

OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    name = "ollama"
    default_model = "llama3"

    def __init__(self, model: str | None = None):
        self.default_model = model or "llama3"

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        start = time.monotonic()
        try:
            resp = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60.0,  # Ollama can be slow
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
            return ProviderResult(
                output=data["response"],
                success=True,
                provider=self.name,
                model=model,
                latency_ms=latency,
            )
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            latency = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                output="",
                success=False,
                provider=self.name,
                model=model,
                latency_ms=latency,
                error_message=str(e),
            )

    def validate(self) -> bool:
        is_valid, _ = self.validate_with_models()
        return is_valid

    def validate_with_models(self) -> tuple[bool, list[str]]:
        try:
            resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if resp.status_code != 200:
                return False, []
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return True, models
        except (httpx.ConnectError, httpx.TimeoutException):
            return False, []
