"""Bridge between DSPy's metric protocol and refinely's registered metrics.

DSPy optimizers call a metric as ``metric(gold, prediction, trace)``. Refinely
scores a run with app-registered `Metric` objects. The bridge turns a DSPy
prediction into a synthetic `Result` and scores it with the exact same metric
set + weight scheme the app registered, so DSPy optimizes against the same
objective as `refinely evaluate`. Token usage and latency come from the
usage-tracking LM wrapper (`refinely.dspy.lm`); when unavailable, the cost and
latency metrics raise `MetricUnavailableError` and drop out of the training
objective (never scored as fake constants).
"""

from typing import Any

from refinely.core.exceptions import EvalError
from refinely.dspy._imports import _dspy
from refinely.dspy.spec import DspyProgramSpec
from refinely.eval.datasets import EvalCase
from refinely.eval.metrics import Metric, MetricUnavailableError, aggregate_scores
from refinely.llm.usage import Result, TokenUsage

CASE_ATTR = "_refinely_case"


def example_case(gold: Any) -> EvalCase:
    """Recover the original `EvalCase` embedded in a prepared example."""
    case = getattr(gold, CASE_ATTR, None)
    if case is None:
        try:
            case = gold[CASE_ATTR]
        except (TypeError, KeyError):
            case = None
    if not isinstance(case, EvalCase):
        raise EvalError(
            f"Prepared example is missing {CASE_ATTR!r}; prepare_example must "
            "embed the original EvalCase so metrics can score it"
        )
    return case


def prediction_result(
    spec: DspyProgramSpec,
    prediction: Any,
    usage: TokenUsage | None = None,
    latency_seconds: float | None = None,
) -> Result:
    """Build a synthetic `Result` from a program prediction."""
    output = spec.prediction_to_output(prediction)
    return Result(
        output=output,
        token_usage=usage,
        latency_seconds=latency_seconds,
    )


def score_result(
    case: EvalCase,
    result: Result,
    metrics: list[Metric],
    weights: dict[str, float],
) -> tuple[dict[str, float], float, str | None]:
    """Score one case like the runner does: metric throw -> 0.0, weighted aggregate.

    Metrics raising `MetricUnavailableError` are excluded from `scores` (and
    therefore from the aggregate, whose remaining weights are renormalized)
    rather than scored as 0.0.

    Returns (scores, aggregate, judge_rationale) where judge_rationale is the
    groundedness judge's one-line rationale when present (feeds the optimizer's
    feedback channel).
    """
    scores: dict[str, float] = {}
    rationale: str | None = None
    for metric in metrics:
        try:
            metric_result = metric.evaluate(case, result)
            scores[metric_result.metric_name] = metric_result.value
            if metric.name == "llm_judge":
                raw = getattr(metric_result, "raw", None)
                if isinstance(raw, dict):
                    rationale = raw.get("rationale")
        except MetricUnavailableError:
            continue  # measurement impossible -> excluded, never scored as 0
        except Exception:  # noqa: BLE001 - mirror runner tolerance
            scores[metric.name] = 0.0
    return scores, aggregate_scores([scores], weights), rationale


def _feedback_text(
    scores: dict[str, float],
    rationale: str | None,
    aggregate: float,
) -> str:
    parts: list[str] = []
    if rationale:
        parts.append(rationale)
    for name, value in sorted(scores.items()):
        if value < 0.5:
            parts.append(f"{name}: {value:.2f}")
    if not parts:
        return f"aggregate: {aggregate:.2f}"
    return "; ".join(parts)


def make_dspy_metric(
    spec: DspyProgramSpec,
    metrics: list[Metric],
    weights: dict[str, float],
):
    """Return a DSPy metric callable scoring `prediction` against `gold`'s case.

    The metric returns `dspy.Prediction(score=aggregate, feedback=...)` so
    optimizers that support feedback (MIPROv2, GEPA) can use it; when dspy is
    not importable it returns the plain float aggregate (dspy-less contexts).
    """

    def metric(gold: Any, prediction: Any, trace: Any | None = None):
        from refinely.dspy.lm import last_latency, last_usage

        case = example_case(gold)
        usage = last_usage()
        latency = last_latency()
        result = prediction_result(spec, prediction, usage=usage, latency_seconds=latency)
        scores, aggregate, rationale = score_result(case, result, metrics, weights)
        feedback = _feedback_text(scores, rationale, aggregate)
        try:
            dspy = _dspy()
        except EvalError:
            return aggregate
        return dspy.Prediction(score=aggregate, feedback=feedback)

    return metric
