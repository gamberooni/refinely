"""The ``config`` command group: manage named app configurations."""

import json

import click
from rich.console import Console

from crucible.config import (
    ConfigError,
    clear_default,
    get_default,
    list_configs,
    rm_config,
    save_config,
    set_default,
    show_config,
)
from crucible.registry import registered_apps

from . import main


@main.group()
def config() -> None:
    """Manage named app configurations stored under configs/<app>/."""


@config.command("save")
@click.argument("name")
@click.option("--app", required=True, type=click.Choice(registered_apps()))
@click.option("--config", required=True, type=str, help="JSON config object to store.")
def config_save(name: str, app: str, config: str) -> None:
    """Save a named config for APP."""
    try:
        parsed = json.loads(config)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid --config JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise click.ClickException("--config must be a JSON object")
    try:
        path = save_config(app, name, parsed)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    Console().print(f"Saved config {name!r} for app {app!r} to {path}")


@config.command("list")
@click.option("--app", default=None, type=click.Choice(registered_apps()))
def config_list(app: str | None) -> None:
    """List named configs, marking the app default with a star."""
    by_app = list_configs(app)
    if not by_app:
        Console().print("No configs found.")
        return
    console = Console()
    for app_name, names in by_app.items():
        default = get_default(app_name)
        for name in names:
            marker = "*" if name == default else " "
            console.print(f"{marker} {app_name}/{name}.json")
        if default is not None:
            console.print(f"  default: {default}")


@config.command("show")
@click.argument("name")
@click.option("--app", required=True, type=click.Choice(registered_apps()))
def config_show(name: str, app: str) -> None:
    """Show the JSON contents of a named config for APP."""
    try:
        contents = show_config(app, name)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    Console().print(json.dumps(contents, indent=2))


@config.command("rm")
@click.argument("name")
@click.option("--app", required=True, type=click.Choice(registered_apps()))
def config_rm(name: str, app: str) -> None:
    """Delete a named config for APP."""
    try:
        rm_config(app, name)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    Console().print(f"Removed config {name!r} for app {app!r}")


@config.command("default")
@click.argument("app", type=click.Choice(registered_apps()))
@click.option(
    "--set", "set_name", default=None, type=str, help="Config name to use as the app default."
)
@click.option("--clear", is_flag=True, help="Clear the app default pointer.")
def config_default(app: str, set_name: str | None, clear: bool) -> None:
    """Set or clear APP's default config used when --config is omitted."""
    if set_name is not None and clear:
        raise click.ClickException("Use either --set <name> or --clear, not both")
    if clear:
        clear_default(app)
        Console().print(f"Cleared default config for app {app!r}")
        return
    if set_name is not None:
        try:
            set_default(app, set_name)
        except ConfigError as exc:
            raise click.ClickException(str(exc)) from exc
        Console().print(f"Default config for app {app!r} set to {set_name!r}")
        return
    current = get_default(app)
    if current is None:
        Console().print(f"No default config set for app {app!r}.")
    else:
        Console().print(f"Default config for app {app!r}: {current}")
