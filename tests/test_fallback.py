from llmx.fallback import get_provider_chain, CATEGORY_RANKINGS


def test_all_categories_have_rankings():
    expected = {"speed", "code", "research", "large_context", "reasoning", "creative"}
    assert set(CATEGORY_RANKINGS.keys()) == expected


def test_get_provider_chain_all_available():
    available = ["groq", "openai", "gemini", "openrouter", "ollama"]
    chain = get_provider_chain("speed", available)
    assert chain == ["groq", "openai", "gemini", "openrouter", "ollama"]


def test_get_provider_chain_some_missing():
    available = ["groq", "ollama"]
    chain = get_provider_chain("code", available)
    # code ranking: openrouter, openai, groq, gemini, ollama
    # only groq and ollama available
    assert chain == ["groq", "ollama"]


def test_get_provider_chain_none_available():
    chain = get_provider_chain("speed", [])
    assert chain == []


def test_code_category_prefers_openrouter():
    available = ["groq", "openai", "gemini", "openrouter", "ollama"]
    chain = get_provider_chain("code", available)
    assert chain[0] == "openrouter"


def test_research_category_prefers_openrouter():
    available = ["groq", "openai", "gemini", "openrouter", "ollama"]
    chain = get_provider_chain("research", available)
    assert chain[0] == "openrouter"
