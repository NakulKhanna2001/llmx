from ..attachments import Attachment
from .base import BaseProvider, ProviderResult
from .chat import request_text

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def answer_text(data: dict) -> str:
    candidate = data["candidates"][0]
    if not isinstance(candidate, dict) or candidate.get("finishReason", "STOP") != "STOP":
        raise ValueError("Gemini did not finish the answer.")
    return "".join(
        part["text"] for part in candidate["content"]["parts"]
        if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
    )


class GeminiProvider(BaseProvider):
    name = "gemini"
    default_model = "gemini-3.5-flash-lite"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        return self.complete_with_attachments(prompt, [], model)

    def complete_with_attachments(
        self, prompt: str, attachments: list[Attachment], model: str | None = None,
    ) -> ProviderResult:
        model = model or self.default_model
        parts = [{"inlineData": {"mimeType": item.mime_type, "data": item.data}} for item in attachments]
        return request_text(
            self.name, model, f"{GEMINI_API_BASE}/{model}:generateContent",
            {"contents": [{"parts": [*parts, {"text": prompt}]}]}, answer_text,
            headers={"x-goog-api-key": self.api_key}, timeout=30.0,
        )

    def validate(self) -> bool:
        return self.complete("Hi").success
