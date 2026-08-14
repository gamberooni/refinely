"""Developer-tooling commands: ``new``, ``doctor``, ``dataset``."""

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from refinely.core.exceptions import EvalError
from refinely.core.settings import Settings
from refinely.devtools.doctor import run_checks
from refinely.devtools.scaffold import ScaffoldError, write_app
from refinely.eval.datasets import dataset_stats
from refinely.registry import registered_apps

from . import context, main
from .context import _format_counts


@main.group()
def new() -> None:
    """Scaffold new apps."""


@new.command("app")
@click.argument("name")
@click.option(
    "--dataset",
    "dataset_path",
    default=None,
    type=click.Path(),
    help="Point the app at an existing dataset instead of writing a stub.",
)
def new_app(name: str, dataset_path: str | None) -> None:
    """Scaffold apps/<NAME>.py and a datasets/<NAME>_v1.json stub."""
    console = Console()
    try:
        app_path, stub_path = write_app(name, Path(dataset_path) if dataset_path else None)
    except ScaffoldError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(f"Created {app_path}")
    if stub_path is not None:
        console.print(f"Created {stub_path}")
    console.print(
        "Register the app by adding to pyproject.toml:\n"
        f'[project.entry-points."refinely.apps"]\n{name} = "apps.{name}"',
        markup=False,
    )


@main.command()
@click.option(
    "--network",
    "network",
    is_flag=True,
    default=False,
    help="Probe the configured gateway/base_url (off by default).",
)
def doctor(network: bool) -> None:
    """Run environment health checks and exit non-zero if any fail."""
    settings = Settings()
    console = Console()
    results = run_checks(settings, network=network)
    for result in results:
        status = "ok" if result.ok else "FAIL"
        lines = [f"[{status}] {result.name}: {result.detail}"]
        if not result.ok and result.hint:
            lines.append(f"      hint: {result.hint}")
        console.print(Panel("\n".join(lines), title=result.name, expand=False))
    if all(r.ok for r in results):
        console.print("all checks passed")
        raise SystemExit(0)
    raise SystemExit(1)


@main.group()
def dataset() -> None:
    """Inspect app datasets."""


@dataset.command()
@click.argument("app", type=click.Choice(registered_apps()))
def stats(app: str) -> None:
    """Show structural statistics for APP's dataset."""
    registration = context.get_registration(app)
    console = Console()
    try:
        stats = dataset_stats(registration.dataset_path)
    except EvalError as exc:
        raise click.ClickException(str(exc)) from exc
    lines = [f"cases:    {stats.case_count}", f"file size: {stats.file_size_bytes} bytes"]
    lines.append("input keys: " + _format_counts(stats.input_field_counts))
    lines.append("expected shapes: " + _format_counts(stats.expected_shape_counts))
    if stats.expected_key_counts:
        lines.append("expected keys: " + _format_counts(stats.expected_key_counts))
    if stats.malformed:
        lines.append(f"malformed cases: {', '.join(stats.malformed)}")
    console.print(
        Panel("\n".join(lines), title=f"dataset: {registration.dataset_path}", expand=False)
    )
