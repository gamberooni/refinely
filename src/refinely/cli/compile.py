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
    "--optimizer",
    default="mipro",
    show_default=True,
    type=click.Choice(["bfs", "mipro"]),
    help="DSPy optimizer: MIPROv2 (instruction-level, default) or BootstrapFewShot (demos).",
)
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
    help="Optimizer max_labeled_demos.",
)
@click.option(
    "--max-bootstrapped-demos",
    default=4,
    show_default=True,
    type=int,
    help="Optimizer max_bootstrapped_demos.",
)
@click.option(
    "--mipro-auto",
    default="light",
    show_default=True,
    type=click.Choice(["light", "medium", "heavy"]),
    help="MIPROv2 budget mode (light = fewest LLM calls).",
)
@click.option(
    "--min-val",
    default=5,
    show_default=True,
    type=int,
    help="Minimum validation cases for the baseline-vs-compiled comparison.",
)
@click.option(
    "--repeats",
    default=3,
    show_default=True,
    type=int,
    help="Repeats of baseline and compiled on validation for the significance gate.",
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
@click.option(
    "--judge-model",
    "judge_model_override",
    default=None,
    type=str,
    help="Model for the LLM judge (default: settings.judge_model or settings.model_name).",
)
def compile(
    app: str,
    optimizer: str,
    max_examples: int | None,
    max_rounds: int,
    max_labeled_demos: int,
    max_bootstrapped_demos: int,
    mipro_auto: str,
    min_val: int,
    repeats: int,
    output_dir: str,
    lineage_db: str | None,
    judge_model_override: str | None,
) -> None:
    """Compile APP's DSPy program and record the baseline-vs-compiled comparison."""
    registration, settings, client, dataset, version = _load_run_context(app)
    settings = (
        settings.model_copy(update={"judge_model": judge_model_override})
        if judge_model_override is not None
        else settings
    )

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
    from refinely.dspy.compile import OPTIMIZER_NAMES, compile_program

    try:
        _dspy()
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Compiling {app!r} with {OPTIMIZER_NAMES[optimizer]} …")
    result = compile_program(
        app_name=app,
        dataset=dataset,
        dataset_version=version,
        client=client,
        settings=settings,
        max_examples=max_examples,
        optimizer=optimizer,
        min_val=min_val,
        repeats=repeats,
        max_rounds=max_rounds,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        mipro_auto=mipro_auto,
        output_dir=output_dir,
    )

    db_path = lineage_db or settings.lineage_db_path
    with LineageDB(db_path) as db:
        compile_id = db.record_compile(
            app_name=app,
            dataset_version=version,
            optimizer=result.optimizer,
            configuration={
                "optimizer": optimizer,
                "max_rounds": max_rounds,
                "max_labeled_demos": max_labeled_demos,
                "max_bootstrapped_demos": max_bootstrapped_demos,
                "mipro_auto": mipro_auto,
                "min_val": min_val,
                "repeats": repeats,
            },
            artifact_path=str(result.artifact_path),
            baseline_score=result.baseline_score,
            compiled_score=result.compiled_score,
            baseline_std=result.baseline_std,
            compiled_std=result.compiled_std,
            verdict=result.verdict,
        )

    verdict_line = (
        "significant" if result.verdict == "significant" else "n.s. — no improvement claim"
    )
    Console().print(
        Panel(
            "\n".join(
                [
                    f"baseline_score:  {result.baseline_score:.4f} ± {result.baseline_std:.4f}",
                    f"compiled_score:  {result.compiled_score:.4f} ± {result.compiled_std:.4f}",
                    f"verdict:         {verdict_line}",
                    f"split:           {result.n_train} train / {result.n_val} validation",
                    f"artifact:        {result.artifact_path}",
                    f"compile recorded: {compile_id}",
                ]
            ),
            title="compile",
        )
    )
