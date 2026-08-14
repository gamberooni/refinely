"""The ``optimize`` command."""

import click
from rich.console import Console
from rich.panel import Panel

from refinely.config import default_config, write_best_config
from refinely.eval.runner import EvaluationRunner
from refinely.optimize.gate import gate_verdict
from refinely.optimize.holdout import split_holdout
from refinely.optimize.objective import build_objective
from refinely.registry import registered_apps
from refinely.tracking.db import LineageDB

from . import context, main
from .context import _load_run_context


@main.command()
@click.argument("app", type=click.Choice(registered_apps()))
@click.option(
    "--trials",
    default=30,
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
    "--judge-model",
    "judge_model_override",
    default=None,
    type=str,
    help="Model for the LLM judge (default: settings.judge_model or settings.model_name).",
)
@click.option(
    "--repeats",
    default=3,
    show_default=True,
    type=int,
    help="Repeats of baseline and best config on validation for the significance gate.",
)
@click.option(
    "--min-val",
    default=3,
    show_default=True,
    type=int,
    help="Minimum validation cases for the holdout split.",
)
@click.option(
    "--tags",
    default=None,
    type=str,
    help="Comma-separated tags recorded on every trial run (e.g. candidate,prod).",
)
def optimize(
    app: str,
    trials: int,
    model_name: str | None,
    judge_model_override: str | None,
    repeats: int,
    min_val: int,
    tags: str | None,
) -> None:
    """Optimize APP's configuration with an Optuna TPE study."""
    if repeats < 2:
        raise click.UsageError(
            "--repeats must be at least 2 (1 makes the significance gate collapse: "
            "std=0 turns any mean gap into a 'significant' result)"
        )
    registration, settings, client, dataset, version = _load_run_context(app)
    console = Console()

    app_settings = settings.model_copy(update={"model_name": model_name or settings.model_name})
    judge_settings = (
        settings.model_copy(update={"judge_model": judge_model_override})
        if judge_model_override is not None
        else settings
    )
    resolved_judge = judge_model_override or settings.judge_model or settings.model_name
    if resolved_judge == app_settings.model_name and judge_model_override is None:
        console.print(
            f"[yellow]warning: judge model {resolved_judge!r} equals the generator model; "
            "pass --judge-model to use a distinct judge[/yellow]"
        )

    search, val = split_holdout(dataset, min_val=min_val)
    console.print(f"holdout: {len(search)} search / {len(val)} validation cases")

    metrics = registration.metrics_factory(client, judge_settings)
    judge = next((m for m in metrics if m.name == "llm_judge"), None)
    judge_model = getattr(judge, "model", None) or judge_settings.judge_model or settings.model_name
    judge_version = getattr(judge, "prompt_version", None)
    app_obj = registration.build_adapter(client, app_settings)
    objective = build_objective(
        app_name=app,
        app=app_obj,
        dataset=search,
        dataset_version=version,
        lineage_db_path=settings.lineage_db_path,
        client=client,
        settings=judge_settings,
        model_name=app_settings.model_name,
        tags=tags,
    )
    study = context.run_study(app, objective, settings.lineage_db_path, n_trials=trials)

    if len(study.trials) == 0 or study.best_trial is None:
        raise click.ClickException(
            "Optimization produced no successful trials; no config was saved."
        )
    best = study.best_trial
    best_config = best.params

    baseline_config = default_config(app, registration.default_config)
    runner = EvaluationRunner(metrics, registration.name, weights=registration.weights)
    baseline_results = [
        runner.run(val, app_obj, config=baseline_config, dataset_version=version)
        for _ in range(repeats)
    ]
    candidate_results = [
        runner.run(val, app_obj, config=best_config, dataset_version=version)
        for _ in range(repeats)
    ]
    gate = gate_verdict(
        [r.aggregate_score for r in baseline_results],
        [r.aggregate_score for r in candidate_results],
    )

    with LineageDB(settings.lineage_db_path) as db:
        for result in baseline_results:
            db.record_run(
                app_name=registration.name,
                dataset_version=version,
                configuration=baseline_config,
                aggregate_score=result.aggregate_score,
                metric_results=result.metric_results,
                case_results=result.case_results,
                weights=registration.weights,
                model_name=app_settings.model_name,
                tags=["gate"],
                judge_model=judge_model if judge is not None else None,
                judge_prompt_version=judge_version,
            )
        for result in candidate_results:
            db.record_run(
                app_name=registration.name,
                dataset_version=version,
                configuration=best_config,
                aggregate_score=result.aggregate_score,
                metric_results=result.metric_results,
                case_results=result.case_results,
                weights=registration.weights,
                model_name=app_settings.model_name,
                tags=["gate"],
                judge_model=judge_model if judge is not None else None,
                judge_prompt_version=judge_version,
            )
        db.record_gate(
            app_name=registration.name,
            trial_number=best.number,
            baseline_mean=gate.baseline.mean,
            baseline_std=gate.baseline.std,
            candidate_mean=gate.candidate.mean,
            candidate_std=gate.candidate.std,
            n_repeats=repeats,
            verdict="significant" if gate.significant else "n.s.",
        )

    if gate.significant:
        path = write_best_config(app, best_config)
        saved_line = f"saved to: {path}"
    else:
        saved_line = "opt-best.json NOT written (n.s.)"
    console.print(
        Panel(
            "\n".join(
                [
                    f"model:            {app_settings.model_name}",
                    f"best trial #{best.number}: aggregate_score = {best.value:.4f}",
                    f"best config:      {best_config}",
                    f"gate (n={repeats} repeats on validation):",
                    f"  baseline:       {gate.baseline.mean:.4f} ± {gate.baseline.std:.4f}",
                    f"  candidate:      {gate.candidate.mean:.4f} ± {gate.candidate.std:.4f}",
                    "verdict:          "
                    + ("significant" if gate.significant else "n.s. — no improvement claim"),
                    saved_line,
                ]
            ),
            title="optimize",
        )
    )
