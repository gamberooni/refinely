"""Click-based command-line interface for crucible."""

import click

from crucible.core.settings import Settings
from crucible.eval.datasets import dataset_version, load_dataset
from crucible.eval.runner import EvaluationRunner
from crucible.llm.client import AsyncOpenAIClient, LLMClient
from crucible.optimize.objective import build_objective
from crucible.optimize.study import run_study
from crucible.registry import discover_apps, get_registration, registered_apps
from crucible.tracking.db import LineageDB

discover_apps()


def _client(settings: Settings) -> LLMClient:
    if not settings.has_api_key:
        raise click.ClickException(
            "CRUCIBLE_OPENAI_API_KEY is not set; set the env var or add a .env file"
        )
    return AsyncOpenAIClient(api_key=settings.openai_api_key, base_url=settings.base_url)


@click.group()
def main() -> None:
    """Crucible: evaluate and optimize LLM application configurations."""


@main.command()
@click.argument("app", type=click.Choice(registered_apps()))
@click.option(
    "--program",
    default=None,
    type=click.Path(exists=True),
    help="Path to a compiled DSPy program JSON to use instead of the LLM client.",
)
def evaluate(app: str, program: str | None) -> None:
    """Run a baseline evaluation of APP against its default dataset."""
    registration = get_registration(app)
    settings = Settings()
    client = _client(settings)
    dataset = load_dataset(registration.dataset_path)
    version = dataset_version(registration.dataset_path)

    if program is not None and registration.dspy_factory is None:
        supporting = ", ".join(
            name for name in registered_apps() if get_registration(name).dspy_factory is not None
        )
        raise click.ClickException(
            f"App {app!r} does not declare a DSPy program. "
            f"Apps that support programs: {supporting or 'none'}"
        )

    app_obj = (
        registration.build_adapter(client, settings, program_path=program)
        if registration.dspy_factory is not None
        else registration.build_adapter(client, settings)
    )
    runner = EvaluationRunner(registration.metrics_factory(client, settings), app)
    result = runner.run(
        dataset,
        app_obj,
        config=registration.default_config,
        dataset_version=version,
    )

    db = LineageDB(settings.lineage_db_path)
    db.init_schema()
    run_id = db.record_run(
        app_name=app,
        dataset_version=version,
        configuration=registration.default_config,
        aggregate_score=result.aggregate_score,
        metric_results=result.metric_results,
        case_results=result.case_results,
        weights=registration.weights,
    )
    db.close()

    click.echo(f"aggregate_score: {result.aggregate_score:.4f}")
    click.echo(f"metric_results: {result.metric_results}")
    click.echo(f"run recorded: {run_id}")


@main.command()
@click.argument("app", type=click.Choice(registered_apps()))
@click.option(
    "--trials",
    default=15,
    show_default=True,
    type=int,
    help="Number of Optuna trials to run.",
)
def optimize(app: str, trials: int) -> None:
    """Optimize APP's configuration with an Optuna TPE study."""
    registration = get_registration(app)
    settings = Settings()
    client = _client(settings)
    dataset = load_dataset(registration.dataset_path)
    version = dataset_version(registration.dataset_path)

    app_obj = registration.build_adapter(client, settings)
    objective = build_objective(
        app_name=app,
        app=app_obj,
        dataset=dataset,
        dataset_version=version,
        lineage_db_path=settings.lineage_db_path,
        client=client,
        settings=settings,
    )
    study = run_study(app, objective, settings.lineage_db_path, n_trials=trials)

    best = study.best_trial
    click.echo(f"best trial #{best.number}: aggregate_score = {best.value:.4f}")
    click.echo(f"best config: {best.params}")


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
    registration = get_registration(app)
    if registration.dspy_factory is None:
        supporting = ", ".join(
            name for name in registered_apps() if get_registration(name).dspy_factory is not None
        )
        raise click.ClickException(
            f"App {app!r} does not declare a DSPy program. "
            f"Apps that support `compile`: {supporting or 'none'}"
        )

    from crucible.dspy._imports import _dspy
    from crucible.dspy.compile import OPTIMIZER_NAME, compile_program

    try:
        _dspy()
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    settings = Settings()
    client = _client(settings)
    dataset = load_dataset(registration.dataset_path)
    version = dataset_version(registration.dataset_path)

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
    db = LineageDB(db_path)
    db.init_schema()
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
    db.close()

    click.echo(f"baseline_score:  {result.baseline_score:.4f}")
    click.echo(f"compiled_score:  {result.compiled_score:.4f}")
    click.echo(f"artifact:        {result.artifact_path}")
    click.echo(f"compile recorded: {compile_id}")


if __name__ == "__main__":
    main()
