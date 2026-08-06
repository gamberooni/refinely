from __future__ import annotations

from typing import Any

from crucible.dspy.spec import DspyProgramSpec
from crucible.eval.datasets import EvalCase
from crucible.llm.usage import Result, TokenUsage


class CompiledProgramAdapter:
    """Wrap a compiled DSPy program as a duck-typed app for EvaluationRunner."""

    def __init__(self, spec: DspyProgramSpec, program: Any) -> None:
        self._spec = spec
        self._program = program

    def execute(self, input: dict, config: dict) -> Result:
        case = EvalCase(id="compiled-program", input=input, expected=None)
        example = self._spec.prepare_example(case)
        prediction = self._program(**dict(example.inputs()))
        output = self._spec.prediction_to_output(prediction)
        return Result(
            output=output,
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
            latency_seconds=0.0,
        )
