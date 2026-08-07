"""Lazy access to the optional `dspy` dependency.

Importing this module never imports dspy; the import happens on the first
`_dspy()` call so refinely remains usable without the `dspy` group installed.
"""

import importlib
from types import ModuleType

from refinely.core.exceptions import EvalError

INSTALL_HINT = "install with `uv sync --group dspy`"


def _dspy() -> ModuleType:
    try:
        return importlib.import_module("dspy")
    except ImportError as exc:
        raise EvalError(f"dspy is not installed; {INSTALL_HINT}") from exc
