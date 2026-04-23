from .groq import GroqProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider
from .openrouter import OpenRouterProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
}
