from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from .config import DEFAULT_CONFIG_PATH, load_config, save_provider_updates
from .providers import PROVIDER_REGISTRY
from .providers.ollama import OllamaProvider

console = Console()

CLOUD_PROVIDERS = {
    "groq": ("Groq", "https://console.groq.com/keys"),
    "openai": ("OpenAI", "https://platform.openai.com/api-keys"),
    "gemini": ("Google Gemini", "https://aistudio.google.com/apikey"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/keys"),
}


def discover_ollama() -> dict:
    reachable, models = OllamaProvider().validate_with_models()
    return {"available": bool(reachable and models), "models": models}


def run_wizard(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    original = load_config(path)
    updates = {}
    console.print("[bold]Configure llmx providers[/bold]")
    console.print("Cloud keys are optional. Enter keeps an existing setting; a dash removes it.")
    console.print("Entering a new key validates it with its provider and may use API credits.")
    for name, (label, url) in CLOUD_PROVIDERS.items():
        console.print(f"\n[bold]{label}[/bold] — {url}")
        key = Prompt.ask("API key", password=True, default="", show_default=False).strip()
        if key == "-":
            updates[name] = None
        elif key:
            if PROVIDER_REGISTRY[name](api_key=key).validate():
                updates[name] = {"api_key": key, "validated": True}
                console.print("Key validated.")
            else:
                console.print("Validation failed. The previous setting was retained.")

    updates["ollama"] = discover_ollama()
    config = save_provider_updates(path, original, updates)
    active = config.get_available_providers()
    console.print("Configured providers:", ", ".join(active) or "none", markup=False)
    console.print(f"Saved configuration to {path}.", markup=False)
    return {"active_providers": active, "config_path": str(path)}
