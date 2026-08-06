"""DSPy compile harness: BootstrapFewShot over an app's program spec + dataset."""

import random
from dataclasses import dataclass
from pathlib import Path

from crucible.core.exceptions import EvalError
from crucible.core.settings import Settings
from crucible.dspy._imports import _dspy
from crucible.dspy.adapter import CompiledProgramAdapter
from crucible.dspy.bridge import make_dspy_metric
from crucible.dspy.lm import configure_lm
from crucible.dspy.spec import DspyProgramSpec
from crucible.eval.datasets import EvalCase
from crucible.eval.metrics import Metric
from crucible.eval.runner import EvaluationRunner
from crucible.llm.client import LLMClient
from crucible.registry import get_registration, registered_apps

TRAIN_FRACTION = 0.7
OPTIMIZER_NAME = "BootstrapFewShot"


@dataclass(frozen=True)
class CompileResult:
    app_name: str
    dataset_version: str
    optimizer: str
    artifact_path: Path
    baseline_score: float
    compiled_score: float
    n_train: int
    n_val: int


def _split_train_val(
    cases: list[EvalCase],
    train_fraction: float = TRAIN_FRACTION,
    max_examples: int | None = None,
    seed: int = 42,
) -> tuple[list[EvalCase], list[EvalCase]]:
    """Shuffle (seeded) and split into train/val, optionally capping total cases."""
    ordered = list(cases)
    if max_examples is not None:
        ordered = ordered[:max_examples]
    if len(ordered) < 2:
        cap = f" after applying max_examples={max_examples}" if max_examples is not None else ""
        raise EvalError(
            "compile needs at least 2 dataset cases for a train/val split"
            f" (got {len(ordered)}{cap})"
        )
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_train = max(1, min(len(ordered) - 1, round(len(ordered) * train_fraction)))
    return ordered[:n_train], ordered[n_train:]


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
    max_rounds: int = 1,
    max_labeled_demos: int = 16,
    max_bootstrapped_demos: int = 4,
    output_dir: str | Path = ".",
    output_name: str = "optimized_program.json",
    seed: int = 42,
) -> CompileResult:
    """Compile an app's DSPy program with BootstrapFewShot and score it on val.

    Baseline (app's registered default config) and compiled (program wrapped in
    `CompiledProgramAdapter`) are both scored through `EvaluationRunner` on the
    same validation split so the artifact's improvement is measurable.
    """
    dspy = _dspy()
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
    spec: DspyProgramSpec = registration.dspy_factory(settings)

    if metrics is None:
        metrics = registration.metrics_factory(client, settings)
    if weights is None:
        weights = registration.weights

    configure_lm(settings, temperature=registration.default_config.get("temperature", 0.0))

    train, val = _split_train_val(dataset, max_examples=max_examples, seed=seed)
    trainset = [spec.prepare_example(case) for case in train]

    runner = EvaluationRunner(metrics, app_name, weights=weights)
    baseline = runner.run(
        val,
        registration.build_adapter(client, settings),
        config=registration.default_config,
        dataset_version=dataset_version,
    )

    program = spec.build()
    dspy_metric = make_dspy_metric(spec, metrics, weights)
    optimizer = dspy.BootstrapFewShot(
        metric=dspy_metric,
        max_rounds=max_rounds,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
    )
    compiled = optimizer.compile(program, trainset=trainset)

    compiled_result = runner.run(
        val,
        CompiledProgramAdapter(spec, compiled),
        config=registration.default_config,
        dataset_version=dataset_version,
    )
    compiled_score = compiled_result.aggregate_score

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / output_name
    compiled.save(str(artifact_path))

    return CompileResult(
        app_name=app_name,
        dataset_version=dataset_version,
        optimizer=OPTIMIZER_NAME,
        artifact_path=artifact_path,
        baseline_score=baseline.aggregate_score,
        compiled_score=compiled_score,
        n_train=len(train),
        n_val=len(val),
    )
