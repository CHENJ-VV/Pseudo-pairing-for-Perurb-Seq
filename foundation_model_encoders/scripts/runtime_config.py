"""Environment-backed configuration helpers for encoder scripts.

The folder-level launcher reads YAML and exports PPFM_* variables before
starting each model script in its selected Python environment. Keeping the
configuration boundary here lets the scientific code remain unchanged while
removing machine-specific paths.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def env_str(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and (value is None or not str(value).strip()):
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Run this script through foundation_model_encoders/launcher.py "
            "or export the variable explicitly."
        )
    return "" if value is None else str(value)


def env_path(
    name: str,
    default: str | Path | None = None,
    *,
    required: bool = False,
    resolve: bool = False,
) -> Path:
    raw_default = None if default is None else str(default)
    value = env_str(name, raw_default, required=required)
    path = Path(value).expanduser()
    return path.resolve() if resolve else path


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, received {value!r}")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default if value is None else value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default if value is None else value)


def env_optional_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(value)


def env_optional_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip()
    if normalized.lower() in {"", "none", "null"}:
        return None
    return normalized


def env_list(name: str, default: Iterable[Any] | None = None) -> list[Any]:
    value = os.environ.get(name)
    if value is None:
        return list(default or [])
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must encode a JSON list or comma-separated values")
    return parsed


def prepend_repo(path: str | Path) -> None:
    """Prepend a repository checkout to sys.path without duplicating entries."""
    import sys

    value = str(Path(path).expanduser())
    if value not in sys.path:
        sys.path.insert(0, value)
