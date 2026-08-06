"""The ``optimize`` command."""

import click
from rich.console import Console
from rich.panel import Panel

from crucible.config import write_best_config
from crucible.optimize.objective import build_objective
from crucible.registry import registered_apps

from . import context, main
from .context import _load_run_context


@main.command()
@click.argument("app", type=click.Choice(registered_apps()))
@click.option(
    "--trials",
    default=15,
    show_default=True,
    type=int,
    help="Number of Optuna trials to run.",
)
@click.option(
    "--model",
    "model_name",
    default=None,
    type=str,
    help="Model name to optimize with (default: settings.model_name). Judge model is unaffected.",
)
@click.option(
    "--tags",
    default=None,
    type=str,
    help="Comma-separated tags recorded on every trial run (e.g. candidate,prod).",
)
def optimize(app: str, trials: int, model_name: str | None, tags: str | None) -> None:
    """Optimize APP's configuration with an Optuna TPE study."""
    registration, settings, client, dataset, version = _load_run_context(app)

    app_settings = settings.model_copy(update={"model_name": model_name or settings.model_name})
    app_obj = registration.build_adapter(client, app_settings)
    objective = build_objective(
        app_name=app,
        app=app_obj,
        dataset=dataset,
        dataset_version=version,
        lineage_db_path=settings.lineage_db_path,
        client=client,
        settings=settings,
        model_name=app_settings.model_name,
        tags=tags,
    )
    study = context.run_study(app, objective, settings.lineage_db_path, n_trials=trials)

    if len(study.trials) == 0 or study.best_trial is None:
        raise click.ClickException(
            "Optimization produced no successful trials; no config was saved."
        )
    best = study.best_trial

    path = write_best_config(app, best.params)

    Console().print(
        Panel(
            "\n".join(
                [
                    f"model: {app_settings.model_name}",
                    f"best trial #{best.number}: aggregate_score = {best.value:.4f}",
                    f"best config: {best.params}",
                    f"saved to: {path}",
                ]
            ),
            title="optimize",
        )
    )
