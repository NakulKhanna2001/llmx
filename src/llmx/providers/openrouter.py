import httpx

from ..fallback import OPENROUTER_CATEGORY_MODELS
from .chat import ChatProvider

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_AUTH_URL = "https://openrouter.ai/api/v1/key"


class OpenRouterProvider(ChatProvider):
    name = "openrouter"
    default_model = "openrouter/free"
    api_url = OPENROUTER_API_URL
    MODEL_FOR_CATEGORY = OPENROUTER_CATEGORY_MODELS

    def model_for_category(self, category: str) -> str:
        return self.MODEL_FOR_CATEGORY.get(category, self.default_model)

    def validate(self) -> bool:
        try:
            response = httpx.get(OPENROUTER_AUTH_URL, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return isinstance(response.json()["data"], dict)
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            return False
