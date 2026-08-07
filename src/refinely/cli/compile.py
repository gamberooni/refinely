"""The ``compile`` command."""

import click
from rich.console import Console
from rich.panel import Panel

from refinely.registry import registered_apps
from refinely.tracking.db import LineageDB

from . import context, main
from .context import _load_run_context


@main.command()
@click.argument("app", type=click.Choice(registered_apps()))
@click.option(
    "--max-examples",
    default=None,
    type=int,
    help="Cap the number of dataset cases used (train + val).",
)
@click.option(
    "--max-rounds",
    default=1,
    show_default=True,
    type=int,
    help="BootstrapFewShot max_rounds.",
)
@click.option(
    "--max-labeled-demos",
    default=16,
    show_default=True,
    type=int,
    help="BootstrapFewShot max_labeled_demos.",
)
@click.option(
    "--max-bootstrapped-demos",
    default=4,
    show_default=True,
    type=int,
    help="BootstrapFewShot max_bootstrapped_demos.",
)
@click.option(
    "--output-dir",
    default=".",
    show_default=True,
    type=click.Path(),
    help="Directory to write the compiled program JSON.",
)
@click.option(
    "--lineage-db",
    default=None,
    type=click.Path(),
    help="SQLite lineage DB path (default: settings.lineage_db_path).",
)
def compile(
    app: str,
    max_examples: int | None,
    max_rounds: int,
    max_labeled_demos: int,
    max_bootstrapped_demos: int,
    output_dir: str,
    lineage_db: str | None,
) -> None:
    """Compile APP's DSPy program with BootstrapFewShot and record results."""
    registration, settings, client, dataset, version = _load_run_context(app)

    if registration.dspy_factory is None:
        supporting = ", ".join(
            name
            for name in registered_apps()
            if context.get_registration(name).dspy_factory is not None
        )
        raise click.ClickException(
            f"App {app!r} does not declare a DSPy program. "
            f"Apps that support `compile`: {supporting or 'none'}"
        )

    from refinely.dspy._imports import _dspy
    from refinely.dspy.compile import OPTIMIZER_NAME, compile_program

    try:
        _dspy()
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Compiling {app!r} with {OPTIMIZER_NAME} …")
    result = compile_program(
        app_name=app,
        dataset=dataset,
        dataset_version=version,
        client=client,
        settings=settings,
        max_examples=max_examples,
        max_rounds=max_rounds,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        output_dir=output_dir,
    )

    db_path = lineage_db or settings.lineage_db_path
    with LineageDB(db_path) as db:
        compile_id = db.record_compile(
            app_name=app,
            dataset_version=version,
            optimizer=result.optimizer,
            configuration={
                "max_rounds": max_rounds,
                "max_labeled_demos": max_labeled_demos,
                "max_bootstrapped_demos": max_bootstrapped_demos,
            },
            artifact_path=str(result.artifact_path),
            baseline_score=result.baseline_score,
            compiled_score=result.compiled_score,
        )

    Console().print(
        Panel(
            "\n".join(
                [
                    f"baseline_score:  {result.baseline_score:.4f}",
                    f"compiled_score:  {result.compiled_score:.4f}",
                    f"artifact:        {result.artifact_path}",
                    f"compile recorded: {compile_id}",
                ]
            ),
            title="compile",
        )
    )
