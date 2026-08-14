"""The ``evaluate`` command."""

import json

import click
from rich.console import Console
from rich.panel import Panel

from refinely.config import ConfigError, default_config, is_valid_name, show_config
from refinely.core.exceptions import EvalError
from refinely.eval.runner import EvaluationRunner
from refinely.registry import AppRegistration, registered_apps
from refinely.tracking.db import LineageDB, _normalize_tags

from . import context, main
from .context import _load_run_context


@main.command()
@click.argument("app", type=click.Choice(registered_apps()))
@click.option(
    "--program",
    default=None,
    type=click.Path(exists=True),
    help="Path to a compiled DSPy program JSON to use instead of the LLM client.",
)
@click.option(
    "--config",
    "config_json",
    default=None,
    type=str,
    help="Config name (under configs/<app>/) or inline JSON object merged onto the app's default config.",
)
@click.option(
    "--model",
    "model_name",
    default=None,
    type=str,
    help="Model name to evaluate with (default: settings.model_name). Judge model is unaffected.",
)
@click.option(
    "--models",
    default=None,
    type=str,
    help="Comma-separated model names to fan out evaluation across (one run per model).",
)
@click.option(
    "--judge-model",
    "judge_model_override",
    default=None,
    type=str,
    help="Model for the LLM judge (default: settings.judge_model or settings.model_name).",
)
@click.option(
    "--judge-consistency",
    is_flag=True,
    help="After the run, re-score a sample with two judge calls and report agreement.",
)
@click.option(
    "--tags",
    default=None,
    type=str,
    help="Comma-separated tags recorded on the run (e.g. candidate,prod).",
)
def evaluate(
    app: str,
    program: str | None,
    config_json: str | None,
    model_name: str | None,
    models: str | None,
    judge_model_override: str | None,
    judge_consistency: bool,
    tags: str | None,
) -> None:
    """Run an evaluation of APP against its default dataset."""
    if model_name is not None and models is not None:
        raise click.ClickException("Use either --model <name> or --models <a,b,c>, not both")

    registration, settings, client, dataset, version = _load_run_context(app)

    if program is not None and registration.dspy_factory is None:
        supporting = ", ".join(
            name
            for name in registered_apps()
            if context.get_registration(name).dspy_factory is not None
        )
        raise click.ClickException(
            f"App {app!r} does not declare a DSPy program. "
            f"Apps that support programs: {supporting or 'none'}"
        )

    config = _resolve_config(app, config_json, registration)

    if models is not None:
        model_list = [m.strip() for m in models.split(",")]
        if not models.strip() or not model_list or any(m == "" for m in model_list):
            raise click.ClickException(
                "--models must be a non-empty comma-separated list of models"
            )
        for model in model_list:
            _run_evaluation(
                registration,
                settings,
                client,
                dataset,
                version,
                config,
                program,
                model=model,
                tags=tags,
                judge_model_override=judge_model_override,
                judge_consistency=judge_consistency,
            )
        return

    _run_evaluation(
        registration,
        settings,
        client,
        dataset,
        version,
        config,
        program,
        model=model_name,
        tags=tags,
        judge_model_override=judge_model_override,
        judge_consistency=judge_consistency,
    )


def _resolve_config(app: str, config_json: str | None, registration: AppRegistration) -> dict:
    """Resolve --config: inline JSON object, named config, or pointer-aware default."""
    if config_json is None:
        try:
            return default_config(app, registration.default_config)
        except ConfigError as exc:
            raise click.ClickException(str(exc)) from exc
    try:
        parsed = json.loads(config_json)
    except json.JSONDecodeError as exc:
        if not is_valid_name(config_json):
            raise click.ClickException(f"Invalid --config JSON: {exc}") from exc
        try:
            named = show_config(app, config_json)
        except ConfigError:
            raise click.ClickException(
                f"Config {config_json!r} not found for app {app!r}; "
                "pass a stored config name or an inline JSON object"
            ) from None
        return {**registration.default_config, **named}
    if not isinstance(parsed, dict):
        raise click.ClickException("--config must be a JSON object")
    return {**registration.default_config, **parsed}


def _run_evaluation(
    registration: AppRegistration,
    settings,
    client,
    dataset: list,
    version: str,
    config: dict,
    program: str | None,
    model: str | None,
    tags: str | None = None,
    judge_model_override: str | None = None,
    judge_consistency: bool = False,
) -> None:
    """Run a single evaluation (app + judge models per D1) and record the run."""
    app_settings = settings.model_copy(update={"model_name": model or settings.model_name})
    judge_settings = (
        settings.model_copy(update={"judge_model": judge_model_override})
        if judge_model_override is not None
        else settings
    )
    console = Console()
    metrics = registration.metrics_factory(client, judge_settings)
    judge = next((m for m in metrics if m.name == "llm_judge"), None)
    judge_model = getattr(judge, "model", None) or judge_settings.judge_model or settings.model_name
    judge_version = getattr(judge, "prompt_version", None)
    if (
        judge is not None
        and judge_model == app_settings.model_name
        and judge_model_override is None
    ):
        console.print(
            f"[yellow]warning: judge model {judge_model!r} equals the generator model; "
            "pass --judge-model to use a distinct judge[/yellow]"
        )
    app_obj = None
    try:
        app_obj = (
            registration.build_adapter(client, app_settings, program_path=program)
            if registration.dspy_factory is not None
            else registration.build_adapter(client, app_settings)
        )
    except EvalError as exc:
        if program is not None:
            raise click.ClickException(f"Failed to load compiled program: {exc}") from exc
        raise
    runner = EvaluationRunner(metrics, registration.name)
    result = runner.run(
        dataset,
        app_obj,
        config=config,
        dataset_version=version,
    )

    if judge_consistency and judge is not None:
        from refinely.eval.datasets import EvalCase
        from refinely.eval.metrics import judge_agreement

        samples = [
            (
                EvalCase(id=cr.case_id, input=cr.input, expected=cr.expected),
                cr.output if isinstance(cr.output, dict) else {},
            )
            for cr in result.case_results[:5]
        ]
        agreement = judge_agreement(client, judge_model, samples)
        console.print(
            f"judge consistency (n={len(samples)}): {agreement:.0%} agreement within tolerance"
        )

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    with LineageDB(settings.lineage_db_path) as db:
        run_id = db.record_run(
            app_name=registration.name,
            dataset_version=version,
            configuration=config,
            aggregate_score=result.aggregate_score,
            metric_results=result.metric_results,
            case_results=result.case_results,
            weights=registration.weights,
            model_name=app_settings.model_name,
            tags=tag_list,
            judge_model=judge_model if judge is not None else None,
            judge_prompt_version=judge_version,
        )

    console.print(
        Panel(
            "\n".join(
                [
                    f"model:            {app_settings.model_name}",
                    f"judge model:      {judge_model if judge is not None else '-'}",
                    f"aggregate_score:  {result.aggregate_score:.4f}",
                    f"metric_results:   {result.metric_results}",
                    f"tags:             {_normalize_tags(tag_list) or '-'}",
                    f"run recorded:     {run_id}",
                ]
            ),
            title="evaluate",
        )
    )
