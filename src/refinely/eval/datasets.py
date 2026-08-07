import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from refinely.core.exceptions import EvalError


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


@dataclass
class DatasetStats:
    """Structural statistics for a dataset file. Never raises on inconsistencies."""

    case_count: int
    file_size_bytes: int
    input_field_counts: dict[str, int]
    expected_shape_counts: dict[str, int]
    expected_key_counts: dict[str, int] = field(default_factory=dict)
    malformed: list[str] = field(default_factory=list)


def _coarse_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (int, float)):
        return "num"
    if value is None:
        return "null"
    return "other"


def dataset_stats(path: str | Path) -> DatasetStats:
    """Compute structural statistics for a dataset file.

    Reuses `load_dataset` for parsing, so structurally invalid files raise the
    same `EvalError` naming the file and offending case. Cases that parse but
    deviate from the modal input-key set or modal expected shape are reported
    in `malformed` by case id.
    """
    p = Path(path)
    cases = load_dataset(p)

    input_keys = Counter()
    expected_shapes = Counter()
    expected_keys = Counter()
    key_sets = Counter()
    for case in cases:
        input_keys.update(case.input.keys())
        key_sets[frozenset(case.input.keys())] += 1
        expected_shapes[_coarse_type(case.expected)] += 1
        if isinstance(case.expected, dict):
            expected_keys.update(case.expected.keys())

    modal_input_keys = set(key_sets.most_common(1)[0][0]) if key_sets else set()
    modal_expected_shape = (
        expected_shapes.most_common(1)[0][0] if expected_shapes else None
    )

    malformed: list[str] = []
    for case in cases:
        if key_sets and set(case.input.keys()) != modal_input_keys:
            malformed.append(case.id)
            continue
        if modal_expected_shape is not None and _coarse_type(case.expected) != modal_expected_shape:
            malformed.append(case.id)

    return DatasetStats(
        case_count=len(cases),
        file_size_bytes=p.stat().st_size if p.exists() else 0,
        input_field_counts=dict(input_keys),
        expected_shape_counts=dict(expected_shapes),
        expected_key_counts=dict(expected_keys) if expected_keys else {},
        malformed=sorted(malformed),
    )
