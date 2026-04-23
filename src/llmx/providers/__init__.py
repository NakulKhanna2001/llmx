from .groq import GroqProvider
from .openai import OpenAIProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
}
