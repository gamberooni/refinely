import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from crucible.core.exceptions import EvalError


class EvalCase(BaseModel):
    id: str
    input: dict[str, Any]
    expected: Any


def load_dataset(path: str | Path) -> list[EvalCase]:
    """Parse a versioned dataset JSON file into `list[EvalCase]`.

    Accepts either a top-level JSON list of cases or a wrapper object
    `{"version": "...", "cases": [...]}`. Raises a clear `EvalError` if any
    case is missing a required field.
    """
    p = Path(path)
    if not p.exists():
        raise EvalError(f"Dataset file not found: {p}")

    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise EvalError(f"Dataset file is not valid JSON: {p}: {e}") from e

    if isinstance(raw, dict) and "cases" in raw:
        raw = raw["cases"]

    if not isinstance(raw, list):
        raise EvalError(f"Dataset file must contain a JSON list of cases: {p}")

    cases: list[EvalCase] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise EvalError(f"Dataset case #{i} in {p} is not a JSON object")
        try:
            cases.append(EvalCase.model_validate(item))
        except ValidationError as e:
            missing = [err["loc"] for err in e.errors() if err["type"] == "missing"]
            raise EvalError(
                f"Dataset case #{i} in {p} is missing required field(s): {missing}"
            ) from e
    return cases


def load_corpus(path: str | Path) -> list[str]:
    """Load the in-memory retrieval corpus from a versioned dataset file."""
    p = Path(path)
    if not p.exists():
        raise EvalError(f"Dataset file not found: {p}")

    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise EvalError(f"Dataset file is not valid JSON: {p}: {e}") from e

    if not isinstance(raw, dict) or "corpus" not in raw:
        raise EvalError(f"Dataset file has no 'corpus' key: {p}")

    corpus = raw["corpus"]
    if not isinstance(corpus, list) or not all(isinstance(s, str) for s in corpus):
        raise EvalError(f"'corpus' in {p} must be a list of strings")
    return corpus


def dataset_version(path: str | Path) -> str:
    """Return the dataset's version string: its `version` key if present, else the filename stem."""
    p = Path(path)
    if p.exists():
        try:
            raw = json.loads(p.read_text())
            if isinstance(raw, dict) and raw.get("version"):
                return str(raw["version"])
        except (json.JSONDecodeError, OSError):
            pass
    return p.stem
