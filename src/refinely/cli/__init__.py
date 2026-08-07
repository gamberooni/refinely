"""Click-based command-line interface for refinely."""

import click

from refinely.registry import discover_apps

from .context import _load_run_context  # noqa: F401


@click.group()
def main() -> None:
    """Refinely: evaluate and optimize LLM application configurations."""


discover_apps()

from . import (  # noqa: E402,F401
    compile,
    config_cmds,
    devtools,
    evaluate,
    optimize,
    readback,
)

__all__ = ["main", "_load_run_context"]
