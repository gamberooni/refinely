"""DSPy integration: compile harness + per-app program declarations.

The `dspy` package is an optional dependency (uv group `dspy`); every
`import dspy` in this package is lazy, routed through `_dspy()`.
"""

from refinely.dspy._imports import _dspy
from refinely.dspy.adapter import CompiledProgramAdapter
from refinely.dspy.bridge import (
    example_case,
    make_dspy_metric,
    prediction_result,
    score_result,
)
from refinely.dspy.compile import CompileResult, compile_program
from refinely.dspy.lm import configure_lm
from refinely.dspy.load import load_program
from refinely.dspy.spec import DspyProgramSpec

__all__ = [
    "CompileResult",
    "CompiledProgramAdapter",
    "DspyProgramSpec",
    "_dspy",
    "compile_program",
    "configure_lm",
    "example_case",
    "load_program",
    "make_dspy_metric",
    "prediction_result",
    "score_result",
]
