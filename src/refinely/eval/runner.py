from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from refinely.eval.datasets import EvalCase
from refinely.eval.metrics import Metric, MetricUnavailableError, aggregate_scores
from refinely.registry import get_registration


class CaseResult(BaseModel):
    case_id: str
    input: dict[str, Any]
    output: dict[str, Any] | str | None
    expected: Any
    scores: dict[str, float]
    error: str | None = None


class EvaluationRunResult(BaseModel):
    app_name: str
    dataset_version: str
    config: dict[str, Any]
    case_results: list[CaseResult]
    metric_results: dict[str, float]
    aggregate_score: float


class EvaluationRunner:
    """Executes every dataset case through an adapter and scores the run."""

    def __init__(
        self,
        metrics: list[Metric],
        app_name: str,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._metrics = metrics
        if weights is None:
            self._weights = get_registration(app_name).weights
        else:
            self._weights = weights
        self._app_name = app_name

    def run(
        self,
        dataset: list[EvalCase],
        app: Any,
        config: dict[str, Any],
        dataset_version: str = "unknown",
    ) -> EvaluationRunResult:
        metric_totals: dict[str, list[float]] = {m.name: [] for m in self._metrics}
        case_results: list[CaseResult] = []

        for case in dataset:
            output: Any = None
            error: str | None = None
            try:
                output = app.execute(case.input, config)
            except Exception as e:  # noqa: BLE001 - tolerate per-case failures
                error = str(e)

            scores: dict[str, float] = {}
            if output is not None:
                for metric in self._metrics:
                    try:
                        result = metric.evaluate(case, output)
                        scores[result.metric_name] = result.value
                        metric_totals.setdefault(result.metric_name, []).append(result.value)
                    except MetricUnavailableError:
                        continue
                    except Exception as e:  # noqa: BLE001 - keep the run going
                        scores[metric.name] = 0.0
                        metric_totals.setdefault(metric.name, []).append(0.0)
                        error = error or f"metric {metric.name} failed: {e}"
            else:
                for metric in self._metrics:
                    scores[metric.name] = 0.0
                    metric_totals.setdefault(metric.name, []).append(0.0)

            case_results.append(
                CaseResult(
                    case_id=case.id,
                    input=case.input,
                    output=output.output if output is not None else None,
                    expected=case.expected,
                    scores=scores,
                    error=error,
                )
            )

        metric_results = {
            name: sum(values) / len(values) for name, values in metric_totals.items() if values
        }
        aggregate = aggregate_scores([c.scores for c in case_results], self._weights)

        return EvaluationRunResult(
            app_name=self._app_name,
            dataset_version=dataset_version,
            config=config,
            case_results=case_results,
            metric_results=metric_results,
            aggregate_score=aggregate,
        )
