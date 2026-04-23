from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

from .config import LlmxConfig, save_config, DEFAULT_CONFIG_PATH
from .providers.groq import GroqProvider
from .providers.openai import OpenAIProvider
from .providers.gemini import GeminiProvider
from .providers.openrouter import OpenRouterProvider
from .providers.ollama import OllamaProvider

console = Console()

PROVIDER_INFO = [
    {
        "name": "groq",
        "label": "Groq",
        "description": "Fast inference for simple tasks",
        "url": "https://console.groq.com/keys",
        "key_prefix": "gsk_",
    },
    {
        "name": "openai",
        "label": "OpenAI",
        "description": "General reasoning, creative writing",
        "url": "https://platform.openai.com/api-keys",
        "key_prefix": "sk-",
    },
    {
        "name": "gemini",
        "label": "Google Gemini",
        "description": "Large context, multimodal",
        "url": "https://aistudio.google.com/apikey",
        "key_prefix": "",
    },
    {
        "name": "openrouter",
        "label": "OpenRouter",
        "description": "DeepSeek (code), Perplexity (research)",
        "url": "https://openrouter.ai/keys",
        "key_prefix": "sk-or-",
    },
]


def run_wizard() -> dict:
    console.print(Panel(
        Text("llmx Setup Wizard", style="bold cyan", justify="center"),
        subtitle="Configure your LLM providers",
    ))
    console.print()

    config = LlmxConfig()
    total = len(PROVIDER_INFO) + 1  # +1 for Ollama
    active_count = 0

    for i, info in enumerate(PROVIDER_INFO, 1):
        console.print(f"[bold][{i}/{total}] {info['label']}[/bold] — {info['description']}")
        console.print(f"       Get a key at: [link]{info['url']}[/link]")

        key = Prompt.ask(
            "       Enter key (or press Enter to skip)",
            default="",
            show_default=False,
        )

        if not key:
            console.print("       [dim]Skipped[/dim]")
            config.providers[info["name"]] = {"api_key": None, "validated": False}
        else:
            console.print("       Validating...", end=" ")
            provider_cls = {
                "groq": GroqProvider,
                "openai": OpenAIProvider,
                "gemini": GeminiProvider,
                "openrouter": OpenRouterProvider,
            }[info["name"]]
            provider = provider_cls(api_key=key)
            valid = provider.validate()

            if valid:
                console.print("[green]Validated[/green]")
                config.providers[info["name"]] = {"api_key": key, "validated": True}
                active_count += 1
            else:
                console.print("[red]Failed[/red] — key will not be used")
                config.providers[info["name"]] = {"api_key": key, "validated": False}

        console.print()

    # Ollama
    console.print(f"[bold][{total}/{total}] Ollama[/bold] — Local fallback, no key needed")
    console.print("       Checking localhost:11434...", end=" ")
    ollama = OllamaProvider()
    is_valid, models = ollama.validate_with_models()

    if is_valid:
        console.print(f"[green]Running[/green] — models: {', '.join(models)}")
        config.providers["ollama"] = {"available": True, "models": models}
        active_count += 1
    else:
        console.print("[yellow]Not running[/yellow]")
        config.providers["ollama"] = {"available": False, "models": []}

    console.print()
    save_config(config)
    console.print(Panel(
        f"[bold green]{active_count}/{total} providers active[/bold green]\n"
        f"Config saved to {DEFAULT_CONFIG_PATH}",
        title="Setup Complete",
    ))

    return {
        "active_providers": config.get_available_providers(),
        "config_path": str(DEFAULT_CONFIG_PATH),
    }
