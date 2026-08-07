"""Click-based command-line interface for refinely."""

import click

from refinely.registry import discover_apps

from .context import _load_run_context


@click.group()
def main() -> None:
    """Refinely: evaluate and optimize LLM application configurations."""


discover_apps()

from . import (  # noqa: F401
    compile,
    config_cmds,
    devtools,
    evaluate,
    optimize,
    readback,
)

__all__ = ["_load_run_context", "main"]
