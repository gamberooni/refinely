"""Bridge between DSPy's metric protocol and refinely's registered metrics.

DSPy optimizers call a metric as ``metric(gold, prediction, trace)``. Refinely
scores a run with app-registered `Metric` objects. The bridge turns a DSPy
prediction into a synthetic `Result` (zero token usage / latency) and scores it
with the exact same metric set + weight scheme the app registered, so DSPy
optimizes against the same objective as `refinely evaluate`.
"""

from typing import Any

from refinely.core.exceptions import EvalError
from refinely.dspy.spec import DspyProgramSpec
from refinely.eval.datasets import EvalCase
from refinely.eval.metrics import Metric, aggregate_scores
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


def prediction_result(spec: DspyProgramSpec, prediction: Any) -> Result:
    """Build a synthetic `Result` from a program prediction (zero usage/latency)."""
    output = spec.prediction_to_output(prediction)
    return Result(
        output=output,
        token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
        latency_seconds=0.0,
    )


def score_result(
    case: EvalCase,
    result: Result,
    metrics: list[Metric],
    weights: dict[str, float],
) -> tuple[dict[str, float], float]:
    """Score one case like the runner does: metric throw -> 0.0, weighted aggregate."""
    scores: dict[str, float] = {}
    for metric in metrics:
        try:
            metric_result = metric.evaluate(case, result)
            scores[metric_result.metric_name] = metric_result.value
        except Exception:  # noqa: BLE001 - mirror runner tolerance
            scores[metric.name] = 0.0
    return scores, aggregate_scores([scores], weights)


def make_dspy_metric(
    spec: DspyProgramSpec,
    metrics: list[Metric],
    weights: dict[str, float],
):
    """Return a DSPy metric callable scoring `prediction` against `gold`'s case."""

    def metric(gold: Any, prediction: Any, trace: Any | None = None) -> float:
        case = example_case(gold)
        result = prediction_result(spec, prediction)
        _, aggregate = score_result(case, result, metrics, weights)
        return aggregate

    return metric
