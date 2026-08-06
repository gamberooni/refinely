from __future__ import annotations

from pathlib import Path
from typing import Any

from crucible.core.settings import Settings
from crucible.dspy._imports import _dspy
from crucible.dspy.lm import configure_lm
from crucible.dspy.spec import DspyProgramSpec


def load_program(
    spec: DspyProgramSpec,
    path: str | Path,
    settings: Settings,
    temperature: float = 0.0,
) -> Any:
    """Build a fresh program, load a compiled artifact, and wire the LM."""
    _dspy()
    program = spec.build()
    program.load(str(path))
    configure_lm(settings, temperature=temperature)
    return program
