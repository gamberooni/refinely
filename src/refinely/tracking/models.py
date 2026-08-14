"""Pydantic models for lineage read-back rows."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvaluationRun(BaseModel):
    """One row of `evaluation_runs` plus its joined metric results."""

    run_id: str
    app_name: str
    dataset_version: str
    configuration: dict[str, Any]
    model_name: str | None = None
    judge_model: str | None = None
    judge_prompt_version: str | None = None
    tags: str | None = None
    optuna_trial_number: int | None = None
    aggregate_score: float
    created_at: str
    metric_results: dict[str, float] = Field(default_factory=dict)


class CompileRecord(BaseModel):
    """One row of `dspy_compiles`."""

    compile_id: str
    app_name: str
    dataset_version: str
    optimizer: str
    configuration: dict[str, Any]
    artifact_path: str
    baseline_score: float
    compiled_score: float
    baseline_std: float | None = None
    compiled_std: float | None = None
    verdict: str | None = None
    created_at: str


class CaseRecord(BaseModel):
    """One row of `case_results` with its per-metric score breakdown."""

    case_id: str
    input: dict[str, Any]
    output: dict[str, Any] | str | None = None
    expected: Any
    score: float
    metric_scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
