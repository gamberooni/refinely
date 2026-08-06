"""Per-app DSPy program declaration.

The spec's callables keep `dspy` imports lazy: building a fresh uncompiled
program, mapping dataset cases to training examples, and mapping predictions
back to the application's output shape all happen inside app modules.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from crucible.eval.datasets import EvalCase


@dataclass(frozen=True)
class DspyProgramSpec:
    """Declares a DSPy program plus the bridges to crucible's world.

    Attributes:
        build: Returns a fresh, uncompiled `dspy.Module` for the app.
        prepare_example: Converts a dataset case into a `dspy.Example` used
            as a gold (training/validation) example.
        prediction_to_output: Maps a program prediction back into the shape
            the app's `execute` produces, so the app's registered metrics
            score compiled output without modification.
    """

    build: Callable[[], Any]
    prepare_example: Callable[[EvalCase], Any]
    prediction_to_output: Callable[[Any], dict[str, Any]]
