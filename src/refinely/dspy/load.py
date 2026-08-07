from __future__ import annotations

from pathlib import Path
from typing import Any

from refinely.core.settings import Settings
from refinely.dspy._imports import _dspy
from refinely.dspy.lm import configure_lm
from refinely.dspy.spec import DspyProgramSpec


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
