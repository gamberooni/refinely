from types import SimpleNamespace

import pytest

from apps.extraction import ExactMatchMetric
from refinely.core.exceptions import EvalError
from refinely.eval.datasets import EvalCase
from refinely.eval.metrics import LatencyMetric, MetricUnavailableError
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


def test_metric_failure_included_in_run_level_mean() -> None:
    """A metric that throws for one case counts 0.0 in the run-level mean too."""

    class _BoomOnA:
        name = "boom"

        def evaluate(self, case: EvalCase, output: Result) -> object:
            if case.id == "a":
                raise RuntimeError("metric exploded on a")
            return SimpleNamespace(metric_name="boom", value=1.0)

    app = _StubApp()
    runner = EvaluationRunner(metrics=[_BoomOnA()], app_name="extraction")

    result = runner.run(_dataset(), app, {})

    # failed case "a" counts 0.0, case "b" scores 1.0 → mean 0.5, matching the aggregate's 0.0 treatment
    assert result.metric_results["boom"] == 0.5


def test_unavailable_metric_excluded_and_renormalized() -> None:
    class _Maybe:
        name = "maybe"

        def evaluate(self, case: EvalCase, output: Result) -> object:
            if case.id == "b":
                raise MetricUnavailableError("usage missing on b")
            return SimpleNamespace(metric_name="maybe", value=1.0)

    class _Always:
        name = "always"

        def evaluate(self, case: EvalCase, output: Result) -> object:
            return SimpleNamespace(metric_name="always", value=1.0)

    runner = EvaluationRunner(
        metrics=[_Always(), _Maybe()],
        app_name="custom_app",
        weights={"always": 0.5, "maybe": 0.5},
    )

    result = runner.run(_dataset(), _StubApp(), {})

    # case a: both score 1.0; case b: "maybe" unavailable -> excluded, "always" renormalized
    assert result.aggregate_score == pytest.approx(1.0)
    # run-level mean is over measured cases only (case a)
    assert result.metric_results["maybe"] == 1.0


def test_metric_unavailable_on_all_cases_is_n_a() -> None:
    class _Never:
        name = "never"

        def evaluate(self, case: EvalCase, output: Result) -> object:
            raise MetricUnavailableError("always unavailable")

    class _Always:
        name = "always"

        def evaluate(self, case: EvalCase, output: Result) -> object:
            return SimpleNamespace(metric_name="always", value=1.0)

    runner = EvaluationRunner(
        metrics=[_Always(), _Never()],
        app_name="custom_app",
        weights={"always": 0.5, "never": 0.5},
    )

    result = runner.run(_dataset(), _StubApp(), {})

    assert "never" not in result.metric_results  # marked n/a by omission
    assert result.aggregate_score == pytest.approx(1.0)  # renormalized over "always"


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
