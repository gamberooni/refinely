"""Locate data files bundled with the installed package (demo datasets).

Demo apps ship with small versioned datasets. In an installed wheel the JSONs
live under ``refinely/datasets/``; in a source checkout they stay at the repo
root ``datasets/``. This helper tries the package location first and falls
back to the repo-relative one so both layouts work.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from refinely.core.exceptions import EvalError


def bundled_dataset(name: str) -> Path:
    """Return a filesystem path to a bundled dataset named ``name``.

    Resolves ``datasets/<name>`` inside the installed ``refinely`` package
    when present, otherwise falls back to ``datasets/<name>`` relative to the
    current working directory (the source-checkout layout).
    """
    pkg = files("refinely").joinpath("datasets", name)
    if pkg.is_file():
        return Path(str(pkg))
    local = Path("datasets") / name
    if local.is_file():
        return local
    raise EvalError(f"Bundled dataset not found: {name!r}")
