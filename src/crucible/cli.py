"""Click-based command-line interface for crucible."""

import math

import click
from rich.console import Console
from rich.panel import Panel

from crucible.core.settings import Settings
from crucible.eval.datasets import EvalCase, dataset_version, load_dataset
from crucible.eval.runner import EvaluationRunner
from crucible.llm.client import AsyncOpenAIClient, LLMClient
from crucible.optimize.objective import build_objective
from crucible.optimize.study import run_study
from crucible.registry import AppRegistration, discover_apps, get_registration, registered_apps
from crucible.reporting.export import export_runs_csv, export_runs_json
from crucible.reporting.render import (
    best_compile_panel,
    best_run_panel,
    cases_table,
    compare_table,
    runs_table,
)
from crucible.tracking.db import LineageDB

discover_apps()


def _client(settings: Settings) -> LLMClient:
    if not settings.has_api_key:
        raise click.ClickException(
            "CRUCIBLE_OPENAI_API_KEY is not set; set the env var or add a .env file"
        )
    return AsyncOpenAIClient(api_key=settings.openai_api_key, base_url=settings.base_url)


def _load_run_context(
    app: str,
) -> tuple[AppRegistration, Settings, LLMClient, list[EvalCase], str]:
    """Build the shared evaluation prelude: registration, settings, client, dataset, version."""
    registration = get_registration(app)
    settings = Settings()
    client = _client(settings)
    dataset = load_dataset(registration.dataset_path)
    version = dataset_version(registration.dataset_path)
    return registration, settings, client, dataset, version


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
    registration, settings, client, dataset, version = _load_run_context(app)

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

    with LineageDB(settings.lineage_db_path) as db:
        run_id = db.record_run(
            app_name=app,
            dataset_version=version,
            configuration=registration.default_config,
            aggregate_score=result.aggregate_score,
            metric_results=result.metric_results,
            case_results=result.case_results,
            weights=registration.weights,
        )

    Console().print(
        Panel(
            "\n".join(
                [
                    f"aggregate_score: {result.aggregate_score:.4f}",
                    f"metric_results: {result.metric_results}",
                    f"run recorded: {run_id}",
                ]
            ),
            title="evaluate",
        )
    )


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
    registration, settings, client, dataset, version = _load_run_context(app)

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
    Console().print(
        Panel(
            "\n".join(
                [
                    f"best trial #{best.number}: aggregate_score = {best.value:.4f}",
                    f"best config: {best.params}",
                ]
            ),
            title="optimize",
        )
    )


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


@main.command()
@click.argument("app", type=click.Choice(registered_apps()))
@click.option(
    "--run",
    "run_id",
    default=None,
    type=str,
    help="Show per-case results for this run id instead of the runs table.",
)
@click.option(
    "--limit",
    default=50,
    show_default=True,
    type=int,
    help="Max number of runs to show.",
)
@click.option(
    "--page",
    default=1,
    show_default=True,
    type=int,
    help="Page of run history to show (1-based, newest first).",
)
@click.option(
    "--pager",
    is_flag=True,
    help="Pipe output through the system pager (less) for scrolling.",
)
def show(app: str, run_id: str | None, limit: int, page: int, pager: bool) -> None:
    """Show APP's run history, best summaries, or a run's per-case results."""
    registration = get_registration(app)
    settings = Settings()
    console = Console()
    with LineageDB(settings.lineage_db_path) as db:
        if run_id is not None:
            if not db.run_exists(run_id):
                raise click.ClickException(f"Run {run_id!r} not found for app {app!r}")
            cases = db.case_results_for_run(run_id)
            if pager:
                with console.pager(styles=True):
                    console.print(cases_table(cases))
            else:
                console.print(cases_table(cases))
            return
        total = db.count_runs(registration.name)
        if total == 0:
            console.print(f"No runs recorded for app {app!r}.")
            return
        if pager:
            runs = db.list_runs(registration.name, limit=total)
            best_run = db.best_run(registration.name)
            best_compile = db.best_compile(registration.name)
            with console.pager(styles=True):
                console.print(runs_table(runs))
                if best_run is not None:
                    console.print(best_run_panel(best_run))
                if best_compile is not None:
                    console.print(best_compile_panel(best_compile))
            return
        offset = (page - 1) * limit
        if offset >= total:
            raise click.ClickException(
                f"Page {page} is out of range for app {app!r} (only {total} runs)."
            )
        runs = db.list_runs(registration.name, limit=limit, offset=offset)
        console.print(runs_table(runs))
        pages = math.ceil(total / limit)
        if pages > 1:
            console.print(f"page {page} of {pages}")
        best_run = db.best_run(registration.name)
        if best_run is not None:
            console.print(best_run_panel(best_run))
        best_compile = db.best_compile(registration.name)
        if best_compile is not None:
            console.print(best_compile_panel(best_compile))


@main.command()
@click.argument("app", type=click.Choice(registered_apps()))
@click.option(
    "--baseline",
    default=None,
    type=str,
    help="Compare every run against this run id instead of the previous run.",
)
@click.option(
    "--page",
    default=1,
    show_default=True,
    type=int,
    help="Page of runs to compare (1-based, chronological).",
)
@click.option(
    "--page-size",
    default=50,
    show_default=True,
    type=int,
    help="Max number of runs per compare page.",
)
@click.option(
    "--pager",
    is_flag=True,
    help="Pipe output through the system pager (less) for scrolling.",
)
def compare(app: str, baseline: str | None, page: int, page_size: int, pager: bool) -> None:
    """Compare APP's runs with per-metric deltas against a baseline."""
    registration = get_registration(app)
    settings = Settings()
    console = Console()
    with LineageDB(settings.lineage_db_path) as db:
        total = db.count_runs(registration.name)
        if total == 0:
            console.print(f"No runs recorded for app {app!r}.")
            return
        if baseline is not None and not db.run_exists(baseline):
            raise click.ClickException(f"Baseline run {baseline!r} not found for app {app!r}")
        baseline_run = db.get_run(baseline) if baseline is not None else None
        if pager:
            runs = db.list_runs(registration.name, limit=total)
            with console.pager(styles=True):
                console.print(
                    compare_table(list(reversed(runs)), baseline_run=baseline_run)
                )
            return
        if (page - 1) * page_size >= total:
            raise click.ClickException(
                f"Page {page} is out of range for app {app!r} (only {total} runs)."
            )
        offset = max(0, total - page * page_size)
        limit = min(page_size, total - (page - 1) * page_size)
        runs = db.list_runs(registration.name, limit=limit, offset=offset)
        console.print(compare_table(list(reversed(runs)), baseline_run=baseline_run))
        pages = math.ceil(total / page_size)
        if pages > 1:
            console.print(f"page {page} of {pages}")


@main.command()
@click.argument("app", type=click.Choice(registered_apps()))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "json"]),
    default="csv",
    show_default=True,
    help="Export format.",
)
@click.option(
    "--output",
    "output",
    default=None,
    type=click.Path(),
    help="Output file path (default: <app>_runs.csv or <app>_runs.json in cwd).",
)
def export(app: str, fmt: str, output: str | None) -> None:
    """Export APP's runs and metric values to a CSV or JSON file."""
    registration = get_registration(app)
    settings = Settings()
    console = Console()
    output_path = output or f"{registration.name}_runs.{fmt}"
    with LineageDB(settings.lineage_db_path) as db:
        runs = db.list_runs(registration.name)
    if fmt == "csv":
        export_runs_csv(runs, output_path)
    else:
        export_runs_json(runs, output_path)
    console.print(f"Wrote {len(runs)} runs to {output_path}")


if __name__ == "__main__":
    main()
