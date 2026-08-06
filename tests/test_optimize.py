import json
import sqlite3

import optuna

from apps.extraction import ExtractionApp
from apps.rag import RAGApp
from crucible.eval.datasets import dataset_version, load_corpus, load_dataset
from crucible.optimize.objective import build_objective
from crucible.optimize.study import run_study
from crucible.tracking.db import LineageDB
from tests.stub_llm import StubLLMClient

EXTRACTION_RESPONSE = {"field_name": "sentiment", "field_value": "positive"}


def _extraction_fixture(tmp_path):
    stub = StubLLMClient(structured_responses=[EXTRACTION_RESPONSE] * 40)
    app = ExtractionApp(stub)
    path = tmp_path / "lineage.db"
    dataset = load_dataset("datasets/extraction_v1.json")
    objective = build_objective(
        app_name="extraction",
        app=app,
        dataset=dataset,
        dataset_version=dataset_version("datasets/extraction_v1.json"),
        lineage_db_path=path,
        client=stub,
    )
    return objective, path


def test_objective_returns_float_and_records_lineage(tmp_path) -> None:
    objective, path = _extraction_fixture(tmp_path)
    trial = optuna.trial.FixedTrial({"temperature": 0.5, "system_prompt_variant": "strict"})

    score = objective(trial)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    db = LineageDB(path)
    db.init_schema()
    assert db.count_runs("extraction") == 1

    conn = sqlite3.connect(path)
    row = conn.execute("SELECT optuna_trial_number, configuration FROM evaluation_runs").fetchone()
    conn.close()
    assert row[0] == 0
    assert json.loads(row[1]) == {
        "temperature": 0.5,
        "system_prompt_variant": "strict",
    }
    db.close()


def test_run_study_runs_n_trials_and_records_all(tmp_path) -> None:
    objective, path = _extraction_fixture(tmp_path)

    study = run_study("extraction", objective, path, n_trials=3)

    assert study.best_trial is not None
    assert 0.0 <= study.best_trial.value <= 1.0

    db = LineageDB(path)
    db.init_schema()
    assert db.count_runs("extraction") == 3

    conn = sqlite3.connect(path)
    numbers = [
        r[0]
        for r in conn.execute(
            "SELECT optuna_trial_number FROM evaluation_runs ORDER BY optuna_trial_number"
        )
    ]
    conn.close()
    assert numbers == [0, 1, 2]
    db.close()


RAG_UNIVERSAL_RESPONSE = {"answer": "Paris", "cited_snippets": [], "scores": [1]}


def _rag_fixture(tmp_path):
    stub = StubLLMClient(
        structured_responses=[RAG_UNIVERSAL_RESPONSE] * 200,
        text_responses=["5"] * 200,
    )
    app = RAGApp(stub, load_corpus("datasets/rag_v1.json"))
    path = tmp_path / "lineage.db"
    dataset = load_dataset("datasets/rag_v1.json")
    objective = build_objective(
        app_name="rag",
        app=app,
        dataset=dataset,
        dataset_version=dataset_version("datasets/rag_v1.json"),
        lineage_db_path=path,
        client=stub,
    )
    return objective, path


def test_run_study_rag_smoke_records_lineage(tmp_path) -> None:
    objective, path = _rag_fixture(tmp_path)

    study = run_study("rag", objective, path, n_trials=2)

    assert study.best_trial is not None
    assert 0.0 <= study.best_trial.value <= 1.0

    db = LineageDB(path)
    db.init_schema()
    assert db.count_runs("rag") == 2
    db.close()


def test_objective_with_explicit_overrides_for_unregistered_app(tmp_path) -> None:
    from crucible.eval.metrics import (
        CostMetric,
        FuzzyMatchMetric,
        LatencyMetric,
        LLMJudgeMetric,
    )

    stub = StubLLMClient(
        structured_responses=[EXTRACTION_RESPONSE] * 40,
        text_responses=["5"] * 40,
    )
    app = ExtractionApp(stub)
    path = tmp_path / "lineage.db"
    dataset = load_dataset("datasets/extraction_v1.json")
    metrics = [
        FuzzyMatchMetric(),
        LLMJudgeMetric(stub, model="gpt-4o-mini"),
        LatencyMetric(),
        CostMetric(),
    ]
    weights = {"fuzzy_match": 0.4, "llm_judge": 0.3, "latency": 0.15, "cost": 0.15}

    def search_space(trial):
        return {
            "temperature": trial.suggest_float("temperature", 0.0, 1.0),
            "system_prompt_variant": trial.suggest_categorical(
                "system_prompt_variant", ["strict", "verbose"]
            ),
        }

    objective = build_objective(
        app_name="custom_app",
        app=app,
        dataset=dataset,
        dataset_version="qa_pairs",
        lineage_db_path=path,
        client=stub,
        metrics=metrics,
        search_space=search_space,
        weights=weights,
    )

    trial = optuna.trial.FixedTrial({"temperature": 0.5, "system_prompt_variant": "strict"})
    score = objective(trial)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    db = LineageDB(path)
    db.init_schema()
    assert db.count_runs("custom_app") == 1
    db.close()
