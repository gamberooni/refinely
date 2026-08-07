"""Per-app named configuration storage on disk.

Named configs live as versionable JSON files under ``configs/<app>/<name>.json``
(cwd-relative, mirroring the ``datasets/`` convention). The per-app default is
a plain-text pointer file ``configs/<app>/.default`` holding a config name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from refinely.core.exceptions import EvalError

CONFIG_DIR = Path("configs")
RESERVED_NAME = "opt-best"
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class ConfigError(EvalError):
    """Raised for invalid config names or missing config files."""


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ConfigError(
            f"Invalid config name {name!r}: must match [A-Za-z0-9_-]+, "
            "with no path separators and no leading dot."
        )
    if name == RESERVED_NAME:
        raise ConfigError(f"Config name {name!r} is reserved and cannot be managed by the user.")


def is_valid_name(name: str) -> bool:
    """Return whether `name` could be a stored config name (used to disambiguate
    a `--config` argument between an inline JSON object and a stored name)."""
    return isinstance(name, str) and bool(_NAME_RE.fullmatch(name))


def config_path(app: str, name: str) -> Path:
    """Return the on-disk path for a named config (without validating the name)."""
    return CONFIG_DIR / app / f"{name}.json"


def save_config(app: str, name: str, config: dict[str, Any]) -> Path:
    """Write a named config as pretty-printed JSON. Returns the written path."""
    _validate_name(name)
    path = config_path(app, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def show_config(app: str, name: str) -> dict[str, Any]:
    """Load and parse a named config. Raises ConfigError if missing or invalid."""
    path = config_path(app, name)
    if not path.exists():
        raise ConfigError(f"Config {name!r} not found for app {app!r} (expected {path})")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {path} must contain a JSON object")
    return raw


def rm_config(app: str, name: str) -> None:
    """Delete a named config. Raises ConfigError if missing; clears the default pointer if needed."""
    path = config_path(app, name)
    if not path.exists():
        raise ConfigError(f"Config {name!r} not found for app {app!r} (expected {path})")
    path.unlink()
    if get_default(app) == name:
        clear_default(app)


def write_best_config(app: str, config: dict[str, Any]) -> Path:
    """Write the optimizer's best config to the reserved `opt-best.json` (overwrites).

    This is the internal writer for the reserved name; user-facing commands cannot
    create, show, or delete `opt-best` via the normal config API.
    """
    path = config_path(app, RESERVED_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def list_configs(app: str | None = None) -> dict[str, list[str]]:
    """Return config names grouped by app, sorted. App directories with no
    configs (or an empty ``configs/`` dir) are omitted."""
    base = CONFIG_DIR
    if not base.exists():
        return {}
    if app is not None:
        app_dir = base / app
        names = sorted(p.stem for p in app_dir.glob("*.json")) if app_dir.exists() else []
        return {app: names}
    result: dict[str, list[str]] = {}
    for app_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        names = sorted(p.stem for p in app_dir.glob("*.json"))
        if names:
            result[app_dir.name] = names
    return result


def _default_path(app: str) -> Path:
    return CONFIG_DIR / app / ".default"


def set_default(app: str, name: str) -> None:
    """Point the app's default at a named config (which must exist)."""
    path = config_path(app, name)
    if not path.exists():
        raise ConfigError(f"Config {name!r} not found for app {app!r} (expected {path})")
    default_path = _default_path(app)
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_text(name + "\n")


def clear_default(app: str) -> None:
    """Clear the app's default pointer, if set."""
    default_path = _default_path(app)
    if default_path.exists():
        default_path.unlink()


def get_default(app: str) -> str | None:
    """Return the app's default config name, or None if unset/empty."""
    default_path = _default_path(app)
    if not default_path.exists():
        return None
    name = default_path.read_text().strip()
    return name or None


def default_config(app: str, registered_default: dict[str, Any]) -> dict[str, Any]:
    """Return the effective config for an app with no explicit `--config`.

    No default pointer → the app's registered default. Pointer set → the named
    config merged over the registered default (so partial configs are safe).
    """
    name = get_default(app)
    if name is None:
        return dict(registered_default)
    return {**registered_default, **show_config(app, name)}
