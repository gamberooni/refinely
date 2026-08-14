"""Seeded search/validation holdout split for the optimize loop."""

import random

from refinely.core.exceptions import EvalError
from refinely.eval.datasets import EvalCase

VAL_FRACTION = 0.3
MIN_VAL_CASES = 3


def split_holdout(
    cases: list[EvalCase],
    val_fraction: float = VAL_FRACTION,
    min_val: int = MIN_VAL_CASES,
    seed: int = 42,
) -> tuple[list[EvalCase], list[EvalCase]]:
    """Shuffle (seeded) and split into (search, val) with val >= min_val cases.

    The search split feeds the Optuna objective; the val split is reserved for
    the final significance gate so the sampler never sees it.
    """
    ordered = list(cases)
    if len(ordered) < min_val + 1:
        raise EvalError(
            f"optimize needs at least {min_val + 1} dataset cases for a holdout "
            f"split (got {len(ordered)})"
        )
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_val = max(min_val, min(len(ordered) - 1, round(len(ordered) * val_fraction)))
    return ordered[: len(ordered) - n_val], ordered[len(ordered) - n_val :]
