"""Statistical gate for the optimize final comparison (baseline vs best trial)."""

import math
from dataclasses import dataclass

# Two-tailed t critical values for a 95% CI, keyed by degrees of freedom (n - 1).
_T_CRITICAL = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}
_FALLBACK_T = 1.96


def _t_critical(n: int) -> float:
    df = max(1, n - 1)
    return _T_CRITICAL.get(df, _FALLBACK_T)


@dataclass(frozen=True)
class GateStats:
    mean: float
    std: float
    n: int
    ci_low: float
    ci_high: float


def _stats(scores: list[float]) -> GateStats:
    n = len(scores)
    if n == 0:
        raise ValueError("gate needs at least one score")
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / max(1, n - 1)
    std = math.sqrt(variance)
    t = _t_critical(n)
    margin = t * std / math.sqrt(n)
    return GateStats(mean=mean, std=std, n=n, ci_low=mean - margin, ci_high=mean + margin)


@dataclass(frozen=True)
class GateResult:
    significant: bool
    baseline: GateStats
    candidate: GateStats


def gate_verdict(baseline_scores: list[float], candidate_scores: list[float]) -> GateResult:
    """True only when the candidate's 95% CI lies entirely above the baseline's.

    Deliberately conservative at small n (t-critical grows as n shrinks), so a
    3-repeat comparison cannot claim an improvement on noise. Refuses n < 2:
    with a single repeat the standard deviation is undefined and the CIs
    collapse to points, making any mean gap look significant.
    """
    if len(baseline_scores) < 2 or len(candidate_scores) < 2:
        raise ValueError(
            "gate needs at least 2 repeats per side (n<2 makes any gap look significant)"
        )
    base = _stats(baseline_scores)
    cand = _stats(candidate_scores)
    return GateResult(
        significant=cand.ci_low > base.ci_high,
        baseline=base,
        candidate=cand,
    )
