import os
import tempfile
from pathlib import Path

from llmx.config import LlmxConfig, load_config, save_config


def test_load_config_missing_file_returns_empty():
    config = load_config(Path("/nonexistent/config.yaml"))
    assert config.providers == {}


def test_save_and_load_roundtrip(tmp_path):
    config_path = tmp_path / "config.yaml"
    config = LlmxConfig(providers={
        "groq": {"api_key": "gsk_test", "validated": True},
        "ollama": {"available": True, "models": ["llama3"]},
    })
    save_config(config, config_path)
    loaded = load_config(config_path)
    assert loaded.providers["groq"]["api_key"] == "gsk_test"
    assert loaded.providers["groq"]["validated"] is True
    assert loaded.providers["ollama"]["models"] == ["llama3"]


def test_get_available_providers():
    config = LlmxConfig(providers={
        "groq": {"api_key": "gsk_test", "validated": True},
        "openai": {"api_key": None, "validated": False},
        "ollama": {"available": True, "models": ["llama3"]},
    })
    available = config.get_available_providers()
    assert "groq" in available
    assert "ollama" in available
    assert "openai" not in available


def test_default_config_path():
    from llmx.config import DEFAULT_CONFIG_PATH
    assert DEFAULT_CONFIG_PATH == Path.home() / ".llmx" / "config.yaml"
