from .groq import GroqProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}
