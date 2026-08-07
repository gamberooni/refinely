"""Read-back commands: ``show``, ``compare``, ``export``."""

import json
import math

import click
from rich.console import Console
from rich.table import Table

from refinely.core.settings import Settings
from refinely.registry import AppRegistration, registered_apps
from refinely.reporting.export import export_runs_csv, export_runs_json
from refinely.reporting.render import (
    best_compile_panel,
    best_run_panel,
    case_pair_summary,
    case_pair_table,
    cases_table,
    compare_table,
    config_delta,
    runs_table,
)
from refinely.tracking.db import LineageDB

from . import context, main
from .context import _resolve_run_id


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
@click.option(
    "--tag",
    "tag",
    default=None,
    type=str,
    help="Restrict to runs recorded with this tag.",
)
def show(app: str, run_id: str | None, limit: int, page: int, pager: bool, tag: str | None) -> None:
    """Show APP's run history, best summaries, or a run's per-case results."""
    registration = context.get_registration(app)
    settings = Settings()
    console = Console()
    with LineageDB(settings.lineage_db_path) as db:
        if run_id is not None:
            resolved = _resolve_run_id(db, app, run_id)
            cases = db.case_results_for_run(resolved)
            errored = sum(1 for case in cases if case.error is not None)
            if pager:
                with console.pager(styles=True):
                    console.print(cases_table(cases))
            else:
                console.print(cases_table(cases))
            if errored:
                console.print(f"{errored} cases errored")
            return
        total = db.count_runs(registration.name)
        if total == 0:
            console.print(f"No runs recorded for app {app!r}.")
            return
        runs = db.list_runs(registration.name, limit=total, tag=tag)
        if not runs:
            console.print("no runs found matching the tag")
            return
        if pager:
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
        runs = db.list_runs(registration.name, limit=limit, offset=offset, tag=tag)
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
@click.option(
    "--model",
    "model_name",
    default=None,
    type=str,
    help="Restrict comparison to runs recorded with this model name.",
)
@click.option(
    "--tag",
    "tag",
    default=None,
    type=str,
    help="Restrict comparison to runs recorded with this tag.",
)
@click.option(
    "--diff-config",
    is_flag=True,
    help="Show key-level configuration deltas against the baseline.",
)
@click.option(
    "--cases",
    "cases_flag",
    is_flag=True,
    help="Compare per-case results between the baseline and the newest run.",
)
def compare(
    app: str,
    baseline: str | None,
    page: int,
    page_size: int,
    pager: bool,
    model_name: str | None,
    tag: str | None,
    diff_config: bool,
    cases_flag: bool,
) -> None:
    """Compare APP's runs with per-metric deltas against a baseline."""
    registration = context.get_registration(app)
    settings = Settings()
    console = Console()
    with LineageDB(settings.lineage_db_path) as db:
        if model_name is not None and db.count_runs(registration.name, model_name=model_name) == 0:
            console.print("no runs found for that model")
            return
        runs = db.list_runs(
            registration.name,
            limit=db.count_runs(registration.name, model_name=model_name),
            model_name=model_name,
            tag=tag,
        )
        if cases_flag:
            _compare_cases(db, registration, runs, baseline, console)
            return
        if diff_config:
            _compare_diff_config(db, registration, runs, baseline, console)
            return
        if not runs:
            if tag is not None:
                console.print("no runs found matching the tag")
            else:
                console.print(f"No runs recorded for app {app!r}.")
            return
        total = len(runs)
        if baseline is not None:
            baseline = _resolve_run_id(db, app, baseline, label="Baseline run")
        baseline_run = db.get_run(baseline) if baseline is not None else None
        if pager:
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
        page_runs = runs[offset : offset + limit]
        console.print(compare_table(list(reversed(page_runs)), baseline_run=baseline_run))
        pages = math.ceil(total / page_size)
        if pages > 1:
            console.print(f"page {page} of {pages}")


def _compare_pair(
    db: LineageDB, app: str, runs: list, baseline: str | None
) -> tuple | None:
    """Resolve (baseline_run, newest_run) for --diff-config/--cases.

    `runs` is newest-first. The baseline is the explicit `--baseline` run when
    given, else the predecessor of the newest run. Returns None when the pair
    cannot be determined (fewer than two runs).
    """
    if len(runs) < 2:
        return None
    newest_run = runs[0]
    if baseline is not None:
        baseline_id = _resolve_run_id(db, app, baseline, label="Baseline run")
        baseline_run = db.get_run(baseline_id)
    else:
        baseline_run = runs[1]
    return baseline_run, newest_run


def _compare_diff_config(
    db: LineageDB,
    registration: AppRegistration,
    runs: list,
    baseline: str | None,
    console: Console,
) -> None:
    """Render a key-level configuration delta between the baseline and newest run."""
    pair = _compare_pair(db, registration.name, runs, baseline)
    if pair is None:
        console.print("comparison needs at least two matching runs")
        return
    baseline_run, newest_run = pair
    delta = config_delta(newest_run.configuration, baseline_run.configuration)
    if not delta:
        console.print("configurations are identical (no changes)")
        return
    table = Table(title="Config delta vs baseline")
    table.add_column("key")
    table.add_column("change")
    table.add_column("before", overflow="fold")
    table.add_column("after", overflow="fold")
    for key, (change, before, after) in delta.items():
        table.add_row(key, change, json.dumps(before, ensure_ascii=False), json.dumps(after, ensure_ascii=False))
    console.print(table)


def _compare_cases(
    db: LineageDB,
    registration: AppRegistration,
    runs: list,
    baseline: str | None,
    console: Console,
) -> None:
    """Render a paired per-case comparison between the baseline and newest run."""
    pair = _compare_pair(db, registration.name, runs, baseline)
    if pair is None:
        console.print("comparison needs at least two matching runs")
        return
    baseline_run, newest_run = pair
    baseline_cases = db.case_results_for_run(baseline_run.run_id)
    newest_cases = db.case_results_for_run(newest_run.run_id)
    if not baseline_cases or not newest_cases:
        console.print("per-case comparison not possible")
        return
    if baseline_run.dataset_version != newest_run.dataset_version:
        console.print(
            f"warning: dataset versions differ ({baseline_run.dataset_version} vs "
            f"{newest_run.dataset_version}); cases are paired by index"
        )
    pairs = []
    for base_case, new_case in zip(baseline_cases, newest_cases):
        delta = None if base_case.score is None or new_case.score is None else new_case.score - base_case.score
        pairs.append((new_case.case_id, base_case.score, new_case.score, delta))
    console.print(case_pair_table(pairs))
    console.print(case_pair_summary(pairs))


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
@click.option(
    "--tag",
    "tag",
    default=None,
    type=str,
    help="Export only runs recorded with this tag.",
)
def export(app: str, fmt: str, output: str | None, tag: str | None) -> None:
    """Export APP's runs and metric values to a CSV or JSON file."""
    registration = context.get_registration(app)
    settings = Settings()
    console = Console()
    output_path = output or f"{registration.name}_runs.{fmt}"
    with LineageDB(settings.lineage_db_path) as db:
        runs = db.list_runs(registration.name, tag=tag)
    if not runs:
        if tag is not None:
            console.print("no runs found matching the tag")
            return
        if fmt == "csv":
            export_runs_csv([], output_path)
        else:
            export_runs_json([], output_path)
        console.print("Wrote 0 runs to " + output_path)
        return
    if fmt == "csv":
        export_runs_csv(runs, output_path)
    else:
        export_runs_json(runs, output_path)
    console.print(f"Wrote {len(runs)} runs to {output_path}")
