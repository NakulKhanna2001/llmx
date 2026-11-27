import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock

DEFAULT_CONFIG_PATH = Path.home() / ".llmx" / "config.yaml"


@dataclass
class LlmxConfig:
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_available_providers(self) -> list[str]:
        available = []
        for name, info in self.providers.items():
            if name == "ollama":
                if info.get("available") and info.get("models"):
                    available.append(name)
            elif info.get("validated") and info.get("api_key"):
                available.append(name)
        return available


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> LlmxConfig:
    if not path.exists():
        return LlmxConfig()
    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError:
        # The YAML parser's diagnostic can include the line containing a key.
        raise ValueError(f"Invalid YAML in {path}. Check the file before running setup.") from None
    if data is None:
        return LlmxConfig()
    providers = data.get("providers", {}) if isinstance(data, dict) else None
    if not isinstance(providers, dict) or not all(
        isinstance(name, str) and isinstance(info, dict) for name, info in providers.items()
    ):
        raise ValueError(f"Expected a mapping of providers in {path}.")
    return LlmxConfig(providers=providers)


def _config_lock(path: Path) -> FileLock:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return FileLock(str(path) + ".lock", timeout=5, mode=0o600)


def _replace_config(config: LlmxConfig, path: Path) -> None:
    # Close the owner-only temp file before rename or cleanup, including on Windows.
    temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    pending = Path(temp.name)
    try:
        with temp as stream:
            yaml.safe_dump({"providers": config.providers}, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def save_config(config: LlmxConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    with _config_lock(path):
        _replace_config(config, path)


def save_provider_updates(path: Path, original: LlmxConfig,
                          updates: dict[str, dict[str, Any] | None]) -> LlmxConfig:
    """Merge only the settings edited by this wizard into the most recent config."""
    with _config_lock(path):
        current = load_config(path)
        for name, value in updates.items():
            saved = current.providers.get(name)
            if saved != original.providers.get(name) and saved != value:
                raise ValueError(f"The {name} setting changed during setup. Run setup again.")
            if value is None:
                current.providers.pop(name, None)
            else:
                current.providers[name] = value
        _replace_config(current, path)
        return current
