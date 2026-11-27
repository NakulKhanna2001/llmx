"""Keep the documented slash command and its registered Python server in sync."""

from importlib.metadata import EntryPoint
import json
from pathlib import Path
import tomllib

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_installed_command_matches_plugin_registration():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    registration = plugin["mcpServers"]["llmx"]
    command = registration["command"]
    entry = EntryPoint(command, project["scripts"][command], "console_scripts")
    assert callable(entry.load())
    assert registration["args"] == ["serve"]


def test_documented_slash_command_is_discoverable():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    skill = (ROOT / "skills" / "llmx" / "SKILL.md").read_text()
    _, header, instructions = skill.split("---", 2)
    name = yaml.safe_load(header)["name"]
    assert f"/{plugin['name']}:{name}" == "/llmx:llmx"
    assert "$ARGUMENTS" in instructions
