"""DSPy integration: compile harness + per-app program declarations.

The `dspy` package is an optional dependency (uv group `dspy`); every
`import dspy` in this package is lazy, routed through `_dspy()`.
"""

from crucible.dspy._imports import _dspy
from crucible.dspy.bridge import (
    example_case,
    make_dspy_metric,
    prediction_result,
    score_result,
)
from crucible.dspy.compile import CompileResult, compile_program
from crucible.dspy.lm import configure_lm
from crucible.dspy.spec import DspyProgramSpec

__all__ = [
    "CompileResult",
    "DspyProgramSpec",
    "_dspy",
    "compile_program",
    "configure_lm",
    "example_case",
    "make_dspy_metric",
    "prediction_result",
    "score_result",
]
