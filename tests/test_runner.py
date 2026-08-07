import pytest

from apps.extraction import ExactMatchMetric
from refinely.core.exceptions import EvalError
from refinely.eval.datasets import EvalCase
from refinely.eval.metrics import LatencyMetric
from refinely.eval.runner import EvaluationRunner
from refinely.llm.usage import Result, TokenUsage


class _StubApp:
    """Deterministic ApplicationAdapter double."""

    def __init__(self, outputs: dict[str, object] | None = None) -> None:
        self._outputs = outputs or {}
        self.fail_cases: set[str] = set()
        self.calls: list[tuple[dict, dict]] = []

    def execute(self, input: dict, config: dict) -> Result:
        self.calls.append((input, config))
        if input.get("_fail") or (input.get("id") in self.fail_cases):
            raise RuntimeError("boom")
        output = self._outputs.get(input.get("id"), input)
        return Result(
            output=output,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            latency_seconds=0.1,
        )


def _dataset() -> list[EvalCase]:
    return [
        EvalCase(id="a", input={"id": "a", "field": "x"}, expected="a"),
        EvalCase(id="b", input={"id": "b", "field": "x"}, expected="b"),
    ]


def test_runner_produces_record_per_case_and_aggregate_score() -> None:
    runner = EvaluationRunner(metrics=[ExactMatchMetric(), LatencyMetric()], app_name="extraction")
    result = runner.run(_dataset(), _StubApp(), {"temperature": 0.0})

    assert len(result.case_results) == 2
    assert {c.case_id for c in result.case_results} == {"a", "b"}
    assert result.aggregate_score > 0.0
    assert result.case_results[0].scores["exact_match"] == 1.0
    assert "latency" in result.metric_results
    assert result.config == {"temperature": 0.0}
    assert result.app_name == "extraction"


def test_runner_executes_every_case() -> None:
    app = _StubApp()
    runner = EvaluationRunner(metrics=[ExactMatchMetric(), LatencyMetric()], app_name="extraction")
    runner.run(_dataset(), app, {})

    assert len(app.calls) == 2


def test_case_execution_failure_does_not_abort_run() -> None:
    app = _StubApp()
    app.fail_cases = {"a"}
    runner = EvaluationRunner(metrics=[ExactMatchMetric(), LatencyMetric()], app_name="extraction")

    result = runner.run(_dataset(), app, {})

    assert len(result.case_results) == 2
    assert result.case_results[0].error is not None
    assert result.case_results[0].output is None
    assert result.case_results[1].error is None
    assert result.case_results[0].scores["exact_match"] == 0.0


def test_metric_failure_does_not_abort_run() -> None:
    class _BoomMetric:
        name = "boom"

        def evaluate(self, case: EvalCase, output: Result) -> object:
            raise RuntimeError("metric exploded")

    app = _StubApp()
    runner = EvaluationRunner(metrics=[ExactMatchMetric(), _BoomMetric()], app_name="extraction")

    result = runner.run(_dataset(), app, {})

    assert len(result.case_results) == 2
    assert result.case_results[0].scores["boom"] == 0.0
    assert result.case_results[0].scores["exact_match"] == 1.0


def test_unknown_app_name_raises() -> None:
    with pytest.raises(EvalError):
        EvaluationRunner(metrics=[], app_name="nope")


def test_explicit_weights_allow_unregistered_app() -> None:
    runner = EvaluationRunner(
        metrics=[ExactMatchMetric(), LatencyMetric()],
        app_name="custom_app",
        weights={"exact_match": 0.8, "latency": 0.2},
    )

    result = runner.run(_dataset(), _StubApp(), {})

    assert result.app_name == "custom_app"
    assert result.aggregate_score > 0.0
