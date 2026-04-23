CATEGORY_RANKINGS: dict[str, list[str]] = {
    "speed":         ["groq", "openai", "gemini", "openrouter", "ollama"],
    "code":          ["openrouter", "openai", "groq", "gemini", "ollama"],
    "large_context": ["gemini", "openai", "openrouter", "groq", "ollama"],
    "research":      ["openrouter", "gemini", "openai", "groq", "ollama"],
    "reasoning":     ["openai", "gemini", "groq", "openrouter", "ollama"],
    "creative":      ["openai", "gemini", "openrouter", "groq", "ollama"],
}

# Category-specific models for OpenRouter
OPENROUTER_CATEGORY_MODELS: dict[str, str] = {
    "code": "deepseek/deepseek-coder",
    "research": "perplexity/sonar",
}


def get_provider_chain(
    category: str,
    available_providers: list[str],
    skip_providers: list[str] | None = None,
) -> list[str]:
    ranking = CATEGORY_RANKINGS.get(category, [])
    skip = set(skip_providers or [])
    return [p for p in ranking if p in available_providers and p not in skip]


def get_model_for_provider(provider: str, category: str) -> str | None:
    if provider == "openrouter":
        return OPENROUTER_CATEGORY_MODELS.get(category)
    return None  # use provider default
