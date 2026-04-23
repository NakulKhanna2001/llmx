from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path.home() / ".llmx" / "config.yaml"


@dataclass
class LlmxConfig:
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_available_providers(self) -> list[str]:
        available = []
        for name, info in self.providers.items():
            if name == "ollama":
                if info.get("available", False):
                    available.append(name)
            elif info.get("validated", False) and info.get("api_key"):
                available.append(name)
        return available


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> LlmxConfig:
    if not path.exists():
        return LlmxConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return LlmxConfig(providers=data.get("providers", {}))


def save_config(config: LlmxConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"providers": config.providers}, f, default_flow_style=False)
