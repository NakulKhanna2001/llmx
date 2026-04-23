from .groq import GroqProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqProvider,
}
