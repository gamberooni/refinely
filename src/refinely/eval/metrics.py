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


class LLMJudgeMetric:
    """1-5 relevance/faithfulness score from an LLM judge call, normalized to [0, 1]."""

    name = "llm_judge"

    def __init__(self, client: LLMClient, model: str) -> None:
        self._client = client
        self._model = model

    def evaluate(self, case: EvalCase, output: Result) -> MetricResult:
        question = str(case.input.get("question", ""))
        actual = _output_text(output)
        expected = _expected_text(case)

        prompt = (
            "Rate the following answer on faithfulness to the expected answer, "
            "from 1 (completely wrong) to 5 (perfect). Respond with ONLY the integer.\n\n"
            f"Question: {question}\n"
            f"Expected: {expected}\n"
            f"Answer: {actual}"
        )
        messages = [
            {"role": "system", "content": "You are a strict evaluation judge."},
            {"role": "user", "content": prompt},
        ]

        try:
            text = asyncio.run(
                self._client.chat_text(self._model, messages, temperature=0.0)
            ).content
            rating = self._parse_rating(text)
        except Exception as e:
            raise EvalError(f"LLM judge failed for case {case.id}: {e}") from e

        return MetricResult(
            metric_name=self.name,
            value=(rating - 1) / 4.0,
            raw={"rating": rating, "judge_text": text},
        )

    @staticmethod
    def _parse_rating(text: str) -> int:
        match = re.search(r"[1-5]", text)
        if not match:
            raise EvalError(f"Judge returned no rating in: {text[:100]!r}")
        return int(match.group(0))


# App-specific weight schemes and metric sets are owned by app modules via
# register_app; the framework only provides the generic metrics library above.


def aggregate_scores(
    case_scores: list[dict[str, float]],
    weights: dict[str, float],
) -> float:
    """Weighted mean of per-case metric scores. Missing metrics count as 0."""
    if not case_scores:
        return 0.0
    totals: list[float] = []
    for scores in case_scores:
        total = sum(weights.get(name, 0.0) * scores.get(name, 0.0) for name in weights)
        totals.append(total)
    return sum(totals) / len(totals)
