"""Small .env reader/writer for local CLI workflows.

The project intentionally avoids adding a dotenv dependency. This parser covers
the simple KEY=VALUE files used by this repo and preserves unrelated lines when
updating known keys.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path


API_BASE_URL_KEYS = ("EMOJI_API_BASE_URL", "OPENAI_BASE_URL", "CHINAMOBILE_MAAS_BASE_URL")
API_KEY_KEYS = ("EMOJI_API_KEY", "OPENAI_API_KEY", "API_KEY", "BAYES_API_KEY")
API_MODEL_KEYS = ("EMOJI_API_MODEL", "OPENAI_MODEL", "VISION_MODEL", "CHINAMOBILE_MAAS_MODEL")


def split_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if value and value[0] in {"'", '"'}:
        try:
            parsed = shlex.split(value, comments=False, posix=True)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = value.strip("'\"")
    else:
        value = value.split(" #", 1)[0].strip()
    return key, value


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = split_env_line(line)
            if parsed is not None:
                key, value = parsed
                values[key] = value
    return values


def load_env_file(path: Path, *, override: bool = False) -> dict[str, str]:
    values = read_env_file(path)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def first_value(values: dict[str, str], keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = os.environ.get(key) or values.get(key)
        if value:
            return value
    return default


def api_settings_from_env(path: Path) -> dict[str, str]:
    values = load_env_file(path)
    return {
        "base_url": first_value(values, API_BASE_URL_KEYS, "https://api.openai.com/v1"),
        "api_key": first_value(values, API_KEY_KEYS, ""),
        "model": first_value(values, API_MODEL_KEYS, "gpt-4o-mini"),
    }


def quote_env_value(value: str) -> str:
    if not value:
        return ""
    if any(char.isspace() or char in "#'\"" for char in value):
        return shlex.quote(value)
    return value


def update_env_file(path: Path, updates: dict[str, str]) -> None:
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    consumed: set[str] = set()
    output: list[str] = []
    for line in existing_lines:
        parsed = split_env_line(line)
        if parsed is None:
            output.append(line)
            continue
        key, _ = parsed
        if key in updates:
            output.append(f"{key}={quote_env_value(updates[key])}")
            consumed.add(key)
        else:
            output.append(line)

    if output and output[-1].strip():
        output.append("")
    for key, value in updates.items():
        if key not in consumed:
            output.append(f"{key}={quote_env_value(value)}")
    path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
