"""DSPy compile harness: MIPROv2 (default) or BootstrapFewShot over an app's program."""

import random
from dataclasses import dataclass
from pathlib import Path

from refinely.core.exceptions import EvalError
from refinely.core.settings import Settings
from refinely.dspy._imports import _dspy
from refinely.dspy.adapter import CompiledProgramAdapter
from refinely.dspy.bridge import make_dspy_metric
from refinely.dspy.lm import configure_lm
from refinely.dspy.spec import DspyProgramSpec
from refinely.eval.datasets import EvalCase
from refinely.eval.metrics import Metric
from refinely.eval.runner import EvaluationRunner
from refinely.llm.client import LLMClient
from refinely.optimize.gate import gate_verdict
from refinely.registry import get_registration, registered_apps

MIN_VAL_CASES = 5
REPEATS = 3
OPTIMIZERS = ("bfs", "mipro")
OPTIMIZER_NAMES = {"bfs": "BootstrapFewShot", "mipro": "MIPROv2"}


@dataclass(frozen=True)
class CompileResult:
    app_name: str
    dataset_version: str
    optimizer: str
    artifact_path: Path
    baseline_score: float
    compiled_score: float
    baseline_std: float
    compiled_std: float
    n_train: int
    n_val: int
    n_repeats: int
    verdict: str


def _split_train_val(
    cases: list[EvalCase],
    train_fraction: float = 0.7,
    max_examples: int | None = None,
    seed: int = 42,
    min_val: int = MIN_VAL_CASES,
) -> tuple[list[EvalCase], list[EvalCase]]:
    """Shuffle (seeded) and split into train/val with val >= min_val cases."""
    ordered = list(cases)
    if max_examples is not None:
        ordered = ordered[:max_examples]
    if len(ordered) < 2 * min_val:
        cap = f" after applying max_examples={max_examples}" if max_examples is not None else ""
        raise EvalError(
            "compile needs at least "
            f"{2 * min_val} dataset cases for a {min_val}-case validation split "
            f"(got {len(ordered)} cases{cap})"
        )
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_val = min(
        max(min_val, round(len(ordered) * (1 - train_fraction))),
        len(ordered) // 2,
    )
    return ordered[: len(ordered) - n_val], ordered[len(ordered) - n_val :]


def _build_optimizer(
    dspy,
    optimizer: str,
    dspy_metric,
    *,
    max_rounds,
    max_labeled_demos,
    max_bootstrapped_demos,
    mipro_auto,
):
    if optimizer == "mipro":
        return dspy.MIPROv2(
            metric=dspy_metric,
            auto=mipro_auto,
            max_labeled_demos=max_labeled_demos,
            max_bootstrapped_demos=max_bootstrapped_demos,
        )
    return dspy.BootstrapFewShot(
        metric=dspy_metric,
        max_rounds=max_rounds,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
    )


def compile_program(
    *,
    app_name: str,
    dataset: list[EvalCase],
    dataset_version: str,
    client: LLMClient,
    settings: Settings | None = None,
    metrics: list[Metric] | None = None,
    weights: dict[str, float] | None = None,
    max_examples: int | None = None,
    optimizer: str = "mipro",
    min_val: int = MIN_VAL_CASES,
    repeats: int = REPEATS,
    max_rounds: int = 1,
    max_labeled_demos: int = 16,
    max_bootstrapped_demos: int = 4,
    mipro_auto: str = "light",
    output_dir: str | Path = ".",
    output_name: str = "optimized_program.json",
    seed: int = 42,
) -> CompileResult:
    """Compile an app's DSPy program and score baseline vs compiled on validation.

    Baseline (app's registered default config) and compiled (program wrapped in
    `CompiledProgramAdapter`) are both scored through `EvaluationRunner` on the
    same validation split with `repeats` repetitions; the aggregate means feed a
    CI-overlap significance gate so the artifact's improvement is measurable.
    """
    if optimizer not in OPTIMIZERS:
        raise EvalError(f"Unknown optimizer {optimizer!r}; choose from {OPTIMIZERS}")
    settings = settings or Settings()
    registration = get_registration(app_name)
    if registration.dspy_factory is None:
        supporting = ", ".join(
            name for name in registered_apps() if get_registration(name).dspy_factory is not None
        )
        raise EvalError(
            f"App {app_name!r} does not declare a DSPy program (dspy_factory); "
            f"only these apps support `compile`: {supporting or 'none'}"
        )
    dspy = _dspy()
    spec: DspyProgramSpec = registration.dspy_factory(settings)

    if metrics is None:
        metrics = registration.metrics_factory(client, settings)
    if weights is None:
        weights = registration.weights

    configure_lm(settings, temperature=registration.default_config.get("temperature", 0.0))

    train, val = _split_train_val(dataset, max_examples=max_examples, seed=seed, min_val=min_val)
    trainset = [spec.prepare_example(case) for case in train]

    # The compiled program does not retrieve, so the training objective excludes
    # retrieval_recall (and the bridge drops cost/latency when usage is absent).
    training_metrics = [m for m in metrics if m.name != "retrieval_recall"]
    training_weights = {k: v for k, v in weights.items() if k != "retrieval_recall"}
    dspy_metric = make_dspy_metric(spec, training_metrics, training_weights)
    optimizer_obj = _build_optimizer(
        dspy,
        optimizer,
        dspy_metric,
        max_rounds=max_rounds,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        mipro_auto=mipro_auto,
    )
    program = spec.build()
    compiled = optimizer_obj.compile(program, trainset=trainset)

    runner = EvaluationRunner(metrics, app_name, weights=weights)
    baseline_adapter = registration.build_adapter(client, settings)
    compiled_adapter = CompiledProgramAdapter(spec, compiled)
    baseline_results = [
        runner.run(
            val,
            baseline_adapter,
            config=registration.default_config,
            dataset_version=dataset_version,
        )
        for _ in range(repeats)
    ]
    compiled_results = [
        runner.run(
            val,
            compiled_adapter,
            config=registration.default_config,
            dataset_version=dataset_version,
        )
        for _ in range(repeats)
    ]
    baseline_scores = [r.aggregate_score for r in baseline_results]
    compiled_scores = [r.aggregate_score for r in compiled_results]
    gate = gate_verdict(baseline_scores, compiled_scores)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / output_name
    compiled.save(str(artifact_path))

    return CompileResult(
        app_name=app_name,
        dataset_version=dataset_version,
        optimizer=OPTIMIZER_NAMES[optimizer],
        artifact_path=artifact_path,
        baseline_score=gate.baseline.mean,
        compiled_score=gate.candidate.mean,
        baseline_std=gate.baseline.std,
        compiled_std=gate.candidate.std,
        n_train=len(train),
        n_val=len(val),
        n_repeats=repeats,
        verdict="significant" if gate.significant else "n.s.",
    )
