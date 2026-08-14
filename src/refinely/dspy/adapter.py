from __future__ import annotations

import time
from typing import Any

from refinely.dspy.spec import DspyProgramSpec
from refinely.eval.datasets import EvalCase
from refinely.llm.usage import Result


class CompiledProgramAdapter:
    """Wrap a compiled DSPy program as a duck-typed app for EvaluationRunner.

    prepare_example receives a synthetic case with expected=None; app
    implementations must tolerate it (rag: non-dict -> {}; extraction:
    non-dict -> field fallback; qa: str(None) in the unused answer slot).
    Token usage comes from the usage-tracking LM wrapper; when the program's
    LM does not surface usage, `token_usage` is None and the cost metric is
    excluded from the run (n/a) rather than scored as zero.
    """

    def __init__(self, spec: DspyProgramSpec, program: Any) -> None:
        self._spec = spec
        self._program = program

    def execute(self, input: dict, config: dict) -> Result:
        from refinely.dspy.lm import last_usage

        case = EvalCase(id="compiled-program", input=input, expected=None)
        example = self._spec.prepare_example(case)
        start = time.perf_counter()
        prediction = self._program(**dict(example.inputs()))
        latency = time.perf_counter() - start
        output = self._spec.prediction_to_output(prediction)
        return Result(
            output=output,
            token_usage=last_usage(),
            latency_seconds=latency,
        )
