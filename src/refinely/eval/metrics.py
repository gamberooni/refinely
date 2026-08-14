import asyncio
import re
from typing import Any, Protocol

from pydantic import BaseModel

from refinely.core.exceptions import EvalError
from refinely.eval.datasets import EvalCase
from refinely.llm.client import LLMClient
from refinely.llm.usage import Result, TokenUsage

# Cost per 1M tokens for the default model (gpt-4o-mini pricing).
PROMPT_PRICE_PER_M = 0.15
COMPLETION_PRICE_PER_M = 0.60

# Latency/cost budgets used to normalize wall-clock and dollar figures to [0, 1].
LATENCY_BUDGET_SECONDS = 10.0
COST_BUDGET_DOLLARS = 0.01


class MetricResult(BaseModel):
    metric_name: str
    value: float
    raw: Any | None = None


class MetricUnavailableError(EvalError):
    """Raised by a metric when its measurement is unavailable for a case.

    The runner treats this distinctly from a metric failure: the metric is
    excluded from the case's scores and the run's means (no fake 0.0 or 1.0),
    so the aggregate and lineage show the metric as n/a rather than wrong.
    """


class Metric(Protocol):
    name: str

    def evaluate(self, case: EvalCase, output: Result) -> MetricResult: ...


def _output_text(output: Result) -> str:
    if isinstance(output.output, str):
        return output.output
    text = str(output.output.get("answer") or "")
    if text:
        return text
    return " ".join(str(v) for v in output.output.values())


def _expected_text(case: EvalCase) -> str:
    expected = case.expected
    if isinstance(expected, dict):
        answer = str(expected.get("answer") or "")
        if answer:
            return answer
    return str(expected)


class FuzzyMatchMetric:
    """Normalized token-overlap / substring match between output and expected text."""

    name = "fuzzy_match"

    def evaluate(self, case: EvalCase, output: Result) -> MetricResult:
        expected = _expected_text(case)
        actual = _output_text(output)
        score = self._score(expected, actual)
        return MetricResult(
            metric_name=self.name,
            value=score,
            raw={"expected": expected, "actual": actual},
        )

    @staticmethod
    def _score(expected: str, actual: str) -> float:
        exp_l = expected.strip().lower()
        act_l = actual.strip().lower()
        if not exp_l:
            return 1.0 if not act_l else 0.0
        if exp_l in act_l:
            return 1.0

        exp_words = set(re.findall(r"[a-z0-9']+", exp_l))
        act_words = set(re.findall(r"[a-z0-9']+", act_l))
        if not exp_words:
            return 0.0
        matched = len(exp_words & act_words) / len(exp_words)

        longest_substring = 0
        for i in range(len(exp_l)):
            for j in range(len(exp_l) - 1, i - 1, -1):
                if exp_l[i : j + 1] in act_l:
                    longest_substring = max(longest_substring, j - i + 1)
                    break
        substring_fraction = longest_substring / len(exp_l)

        return max(matched, substring_fraction)


class LatencyMetric:
    """Wall-clock duration per case, normalized to [0, 1]."""

    name = "latency"

    def __init__(self, budget_seconds: float = LATENCY_BUDGET_SECONDS) -> None:
        self._budget = budget_seconds

    def evaluate(self, case: EvalCase, output: Result) -> MetricResult:
        if output.latency_seconds is None:
            raise MetricUnavailableError(f"latency unavailable for case {case.id}")
        latency = output.latency_seconds
        score = max(0.0, 1.0 - latency / self._budget)
        return MetricResult(
            metric_name=self.name,
            value=score,
            raw={"latency_seconds": latency},
        )


class CostMetric:
    """Estimated USD cost from token usage, normalized to [0, 1]."""

    name = "cost"

    def __init__(
        self,
        prompt_price_per_m: float = PROMPT_PRICE_PER_M,
        completion_price_per_m: float = COMPLETION_PRICE_PER_M,
        budget_dollars: float = COST_BUDGET_DOLLARS,
    ) -> None:
        self._prompt_price = prompt_price_per_m
        self._completion_price = completion_price_per_m
        self._budget = budget_dollars

    def evaluate(self, case: EvalCase, output: Result) -> MetricResult:
        if output.token_usage is None:
            raise MetricUnavailableError(f"cost unavailable for case {case.id}")
        usage: TokenUsage = output.token_usage
        cost = (
            usage.prompt_tokens * self._prompt_price
            + usage.completion_tokens * self._completion_price
        ) / 1_000_000
        score = max(0.0, min(1.0, 1.0 - cost / self._budget))
        return MetricResult(
            metric_name=self.name,
            value=score,
            raw={"cost_dollars": cost},
        )


class JudgeScore(BaseModel):
    """Structured output of the groundedness judge."""

    score: float
    rationale: str


class LLMJudgeMetric:
    """Context-grounded 0-1 faithfulness/completeness score from an LLM judge call.

    The prompt contains the question, the answer, and the retrieved context —
    never the expected answer — so the judge measures grounding in the context,
    disjoint from the deterministic ``fuzzy_match`` (gold-overlap) metric.
    """

    name = "llm_judge"
    prompt_version = "groundedness-v1"

    def __init__(self, client: LLMClient, model: str, temperature: float = 0.0) -> None:
        self._client = client
        self.model = model
        self._temperature = temperature

    def evaluate(self, case: EvalCase, output: Result) -> MetricResult:
        question = str(case.input.get("question", ""))
        actual = _output_text(output)
        context = _context_text(output)
        if not context:
            raise EvalError(
                f"LLM judge has no context for case {case.id}; the app output "
                "must include 'context' (retrieved snippets) for grounding"
            )

        prompt = (
            "Score the answer's groundedness in the provided context.\n"
            "Assess (1) faithfulness — is every claim in the answer supported "
            "by the context? — and (2) completeness — does the answer address "
            "the question given the context? Respond with a score between 0.0 "
            "and 1.0 and a one-line rationale.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer: {actual}"
        )
        messages = [
            {"role": "system", "content": "You are a strict groundedness judge."},
            {"role": "user", "content": prompt},
        ]

        try:
            completion = asyncio.run(
                self._client.chat_structured(
                    self.model, messages, JudgeScore, temperature=self._temperature
                )
            )
        except Exception as e:
            raise EvalError(f"LLM judge failed for case {case.id}: {e}") from e

        score = max(0.0, min(1.0, float(completion.content.score)))
        return MetricResult(
            metric_name=self.name,
            value=score,
            raw={"rationale": completion.content.rationale},
        )


def _context_text(output: Result) -> str:
    if isinstance(output.output, dict):
        return str(output.output.get("context") or "")
    return ""


def judge_agreement(
    client: LLMClient,
    model: str,
    samples: list[tuple[EvalCase, dict]],
    temperature: float = 0.7,
    tolerance: float = 0.25,
) -> float:
    """Re-score samples with two judge calls at nonzero temperature; return the
    fraction of cases whose two scores agree within tolerance."""
    judge = LLMJudgeMetric(client=client, model=model, temperature=temperature)
    total = 0
    agreements = 0
    for case, output_dict in samples:
        result = Result(
            output=output_dict,
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
            latency_seconds=0.0,
        )
        try:
            a = judge.evaluate(case, result).value
            b = judge.evaluate(case, result).value
        except EvalError:
            continue
        total += 1
        if abs(a - b) <= tolerance:
            agreements += 1
    return agreements / total if total else 1.0


# App-specific weight schemes and metric sets are owned by app modules via
# register_app; the framework only provides the generic metrics library above.


def aggregate_scores(
    case_scores: list[dict[str, float]],
    weights: dict[str, float],
) -> float:
    """Weighted mean of per-case metric scores.

    Metrics missing from a case's scores are treated as *unavailable* (the
    measurement was not possible, as opposed to a failed measurement which the
    runner records as 0.0): they are excluded and the remaining weights are
    renormalized to sum to 1.0, so an aggregate is always a proper weighted
    mean of what was actually measured. Cases with no measurable metrics are
    excluded from the run mean.
    """
    if not case_scores:
        return 0.0
    totals: list[float] = []
    for scores in case_scores:
        active = {name: w for name, w in weights.items() if name in scores}
        weight_sum = sum(active.values())
        if not active or weight_sum == 0.0:
            continue
        totals.append(sum(w * scores[name] for name, w in active.items()) / weight_sum)
    return sum(totals) / len(totals) if totals else 0.0
