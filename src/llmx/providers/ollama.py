import os

import httpx

from .base import BaseProvider, ProviderResult
from .chat import request_text

OLLAMA_BASE_URL = "http://localhost:11434"


def generation_text(data: dict) -> str:
    if data["done"] is not True or data.get("done_reason", "stop") != "stop":
        raise ValueError("Incomplete generation")
    return data["response"]


class OllamaProvider(BaseProvider):
    name = "ollama"
    default_model = "llama3"

    def __init__(self, model: str | None = None):
        self.default_model = model or self.default_model
        host = os.environ.get("OLLAMA_HOST", "").strip().rstrip("/") or OLLAMA_BASE_URL
        self.base_url = host if "://" in host else f"http://{host}"

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        model = model or self.default_model
        return request_text(
            self.name, model, f"{self.base_url}/api/generate",
            {"model": model, "prompt": prompt, "stream": False},
            generation_text, timeout=60.0,
        )

    def validate(self) -> bool:
        available, _ = self.validate_with_models()
        return available

    def validate_with_models(self) -> tuple[bool, list[str]]:
        models = []
        try:
            with httpx.Client(base_url=self.base_url, timeout=5.0) as client:
                tags = client.get("/api/tags")
                tags.raise_for_status()
                items = tags.json()["models"]
                if not isinstance(items, list):
                    return False, []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    if not isinstance(name, str) or not name.strip() or name in models:
                        continue
                    if self._supports_local_completion(client, item):
                        models.append(name)
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, TypeError, KeyError):
            return False, []
        return bool(models), models

    @staticmethod
    def _supports_local_completion(client: httpx.Client, model: dict) -> bool:
        if model.get("remote_model") or model.get("remote_host"):
            return False
        capabilities = model.get("capabilities")
        if isinstance(capabilities, list):
            return "completion" in capabilities

        # Older servers report capabilities only in the model details.
        try:
            response = client.post("/api/show", json={"model": model["name"]})
            response.raise_for_status()
            details = response.json()
            return (
                isinstance(details, dict)
                and not details.get("remote_model")
                and not details.get("remote_host")
                and isinstance(details.get("capabilities"), list)
                and "completion" in details["capabilities"]
            )
        except (httpx.HTTPError, ValueError):
            return False
