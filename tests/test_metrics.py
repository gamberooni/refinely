import pytest

from refinely.core.exceptions import EvalError
from refinely.eval.datasets import EvalCase
from refinely.eval.metrics import (
    CostMetric,
    FuzzyMatchMetric,
    LatencyMetric,
    LLMJudgeMetric,
    aggregate_scores,
    judge_agreement,
)
from refinely.llm.usage import Result, TokenUsage
from tests.stub_llm import StubLLMClient


def _case(expected: object, question: str | None = None) -> EvalCase:
    input_data = {"question": question} if question else {"text": "some input"}
    return EvalCase(id="c1", input=input_data, expected=expected)


def _result(output: object, latency: float = 0.5, tokens: tuple[int, int] = (100, 20)) -> Result:
    return Result(
        output=output,
        token_usage=TokenUsage(prompt_tokens=tokens[0], completion_tokens=tokens[1]),
        latency_seconds=latency,
    )


def test_fuzzy_match_exact_substring() -> None:
    m = FuzzyMatchMetric().evaluate(
        _case("Paris"), _result({"answer": "The capital of France is Paris"})
    )
    assert m.value == 1.0


def test_fuzzy_match_partial_overlap() -> None:
    m = FuzzyMatchMetric().evaluate(
        _case("embedded SQL database engine"),
        _result({"answer": "SQLite is an embedded database"}),
    )
    assert 0.0 < m.value < 1.0


def test_fuzzy_match_no_overlap() -> None:
    m = FuzzyMatchMetric().evaluate(_case("Tokyo"), _result({"answer": "Paris"}))
    assert m.value == 0.0


def test_fuzzy_match_uses_answer_key_of_dict_expected() -> None:
    expected = {"answer": "The capital of France is Paris", "source_indices": [0, 1]}
    m = FuzzyMatchMetric().evaluate(
        _case(expected),
        _result(
            {
                "answer": "The capital of France is Paris",
                "retrieved_indices": [0, 1],
                "cited_indices": [0],
            }
        ),
    )
    assert m.value == 1.0


def test_latency_metric_normalizes() -> None:
    fast = LatencyMetric(budget_seconds=10.0).evaluate(_case("x"), _result("y", latency=0.0))
    slow = LatencyMetric(budget_seconds=10.0).evaluate(_case("x"), _result("y", latency=20.0))
    assert fast.value == 1.0
    assert slow.value == 0.0


def test_cost_metric_normalizes() -> None:
    cheap = CostMetric().evaluate(_case("x"), _result("y", tokens=(10, 5)))
    assert cheap.value > 0.99
    assert cheap.raw["cost_dollars"] >= 0.0


def test_llm_judge_groundedness_score() -> None:
    stub = StubLLMClient(structured_responses=[{"score": 1.0, "rationale": "fully supported"}])
    m = LLMJudgeMetric(client=stub, model="judge-1").evaluate(
        _case("Paris", question="capital?"),
        _result({"answer": "Paris", "context": "Paris is the capital of France."}),
    )
    assert m.value == 1.0
    assert stub.structured_calls[0]["model"] == "judge-1"


def test_llm_judge_prompt_does_not_leak_expected_answer() -> None:
    stub = StubLLMClient(structured_responses=[{"score": 0.8, "rationale": "ok"}])
    LLMJudgeMetric(client=stub, model="judge-1").evaluate(
        _case("Rome", question="capital of France?"),
        _result({"answer": "Paris", "context": "Paris is the capital of France."}),
    )
    prompt = stub.structured_calls[0]["messages"][-1]["content"]
    assert "Rome" not in prompt
    assert "Expected" not in prompt
    assert "Context:" in prompt


def test_llm_judge_requires_context() -> None:
    stub = StubLLMClient(structured_responses=[{"score": 0.5, "rationale": "x"}])
    with pytest.raises(EvalError):
        LLMJudgeMetric(client=stub, model="judge-1").evaluate(
            _case("Paris", question="capital?"), _result({"answer": "Paris"})
        )


def test_llm_judge_clamps_score() -> None:
    stub = StubLLMClient(structured_responses=[{"score": 1.7, "rationale": "over"}])
    m = LLMJudgeMetric(client=stub, model="judge-1").evaluate(
        _case("Paris", question="capital?"),
        _result({"answer": "Paris", "context": "Paris is the capital."}),
    )
    assert m.value == 1.0


def test_judge_agreement_reports_fraction() -> None:
    stub = StubLLMClient(
        structured_responses=[
            {"score": 0.9, "rationale": "a"},
            {"score": 0.9, "rationale": "b"},
            {"score": 0.1, "rationale": "c"},
            {"score": 0.9, "rationale": "d"},
        ]
    )
    samples = [
        (_case("P1", question="q1"), {"answer": "a1", "context": "ctx"}),
        (_case("P2", question="q2"), {"answer": "a2", "context": "ctx"}),
    ]
    assert judge_agreement(stub, "judge-1", samples) == 0.5


def test_aggregate_scores_weighted_mean() -> None:
    scores = [
        {"exact_match": 1.0, "latency": 1.0, "cost": 1.0},
        {"exact_match": 0.0, "latency": 1.0, "cost": 1.0},
    ]
    weights = {"exact_match": 0.7, "latency": 0.15, "cost": 0.15}
    expected = (1.0 * 1.0 + 0.0 * 0.7 + 1.0 * 0.15 + 1.0 * 0.15) / 2
    assert aggregate_scores(scores, weights) == pytest.approx(expected)


def test_aggregate_scores_missing_metric_excluded_and_renormalized() -> None:
    scores = [{"exact_match": 0.5}]
    weights = {"exact_match": 0.5, "latency": 0.5}
    # latency unavailable -> excluded, exact_match weight renormalized to 1.0
    assert aggregate_scores(scores, weights) == pytest.approx(0.5)


def test_aggregate_scores_renormalizes_per_case() -> None:
    scores = [
        {"exact_match": 1.0},  # latency unavailable for this case
        {"exact_match": 0.0, "latency": 1.0},
    ]
    weights = {"exact_match": 0.5, "latency": 0.5}
    # case 1: 1.0 (renormalized), case 2: 0.5*0 + 0.5*1 = 0.5 -> mean 0.75
    assert aggregate_scores(scores, weights) == pytest.approx(0.75)


def test_aggregate_scores_skips_fully_unavailable_cases() -> None:
    scores = [{}, {"exact_match": 1.0}]
    weights = {"exact_match": 1.0}
    assert aggregate_scores(scores, weights) == pytest.approx(1.0)


def test_aggregate_scores_empty() -> None:
    assert aggregate_scores([], {"exact_match": 1.0}) == 0.0
