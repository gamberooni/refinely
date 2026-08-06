import sqlite3

import pytest

from apps.extraction import EXTRACTION_WEIGHTS
from apps.qa import QA_WEIGHTS
from crucible.eval.runner import CaseResult
from crucible.tracking.db import LineageDB


def _case_results(n: int = 3) -> list[CaseResult]:
    return [
        CaseResult(
            case_id=f"c{i}",
            input={"text": f"input {i}"},
            output={"field_value": "positive"},
            expected={"field_name": "sentiment", "field_value": "positive"},
            scores={"exact_match": 1.0 if i == 0 else 0.0, "latency": 1.0, "cost": 1.0},
        )
        for i in range(n)
    ]


def test_schema_initializes_fresh_and_is_noop_on_existing(
    tmp_path: pytest.TempPathFactory,
) -> None:
    db_path = tmp_path / "lineage.db"
    db = LineageDB(db_path)
    db.init_schema()
    db.init_schema()  # second call must not error
    db.close()

    conn = sqlite3.connect(db_path)
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('evaluation_runs', 'metric_results', 'case_results', 'dspy_compiles')"
        )
    }
    conn.close()
    assert tables == {
        "evaluation_runs",
        "metric_results",
        "case_results",
        "dspy_compiles",
    }


def test_schema_coexists_with_optuna_tables(tmp_path: pytest.TempPathFactory) -> None:
    db_path = tmp_path / "shared.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE trials (trial_id INTEGER)")
    conn.commit()
    conn.close()

    db = LineageDB(db_path)
    db.init_schema()  # must not clobber Optuna's table
    db.close()

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='trials'").fetchone()
    conn.close()


def test_record_run_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    config = {"temperature": 0.3, "system_prompt_variant": "strict"}
    run_id = db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration=config,
        aggregate_score=0.85,
        metric_results={"exact_match": 0.66, "latency": 1.0, "cost": 1.0},
        case_results=_case_results(),
        weights=EXTRACTION_WEIGHTS,
        optuna_trial_number=7,
    )

    assert isinstance(run_id, str) and run_id
    best = db.best_run("extraction")
    assert best is not None
    assert best["configuration"] == config
    assert best["aggregate_score"] == pytest.approx(0.85)
    assert best["optuna_trial_number"] == 7
    assert best["dataset_version"] == "extraction_v1"

    cases = db.case_results_for_run(run_id)
    assert len(cases) == 3
    by_id = {c["case_id"]: c for c in cases}
    assert by_id["c0"]["input"] == {"text": "input 0"}
    assert by_id["c0"]["output"] == {"field_value": "positive"}
    assert by_id["c0"]["expected"] == {
        "field_name": "sentiment",
        "field_value": "positive",
    }
    assert by_id["c0"]["score"] == pytest.approx(0.7 * 1.0 + 0.15 * 1.0 + 0.15 * 1.0)
    db.close()


def test_best_run_orders_by_score(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    for score in (0.4, 0.9, 0.6):
        db.record_run(
            app_name="qa",
            dataset_version="qa_v1",
            configuration={"temperature": 0.1},
            aggregate_score=score,
            metric_results={"fuzzy_match": score},
            case_results=_case_results(1),
            weights=QA_WEIGHTS,
        )

    best = db.best_run("qa")
    assert best is not None
    assert best["aggregate_score"] == pytest.approx(0.9)

    assert db.best_run("extraction") is None
    db.close()


def test_case_results_ordered_by_score_ascending(
    tmp_path: pytest.TempPathFactory,
) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    cases = _case_results(3)
    cases[0].scores = {"exact_match": 0.0, "latency": 1.0, "cost": 1.0}
    cases[1].scores = {"exact_match": 1.0, "latency": 1.0, "cost": 1.0}
    cases[2].scores = {"exact_match": 0.0, "latency": 0.0, "cost": 0.0}
    run_id = db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration={},
        aggregate_score=0.5,
        metric_results={},
        case_results=cases,
        weights=EXTRACTION_WEIGHTS,
    )

    result = db.case_results_for_run(run_id)
    assert [c["case_id"] for c in result] == ["c2", "c0", "c1"]
    db.close()


def test_record_compile_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    config = {"max_rounds": 1, "max_labeled_demos": 16}
    compile_id = db.record_compile(
        app_name="extraction",
        dataset_version="extraction_v1",
        optimizer="BootstrapFewShot",
        configuration=config,
        artifact_path="optimized_program.json",
        baseline_score=0.6,
        compiled_score=0.9,
    )

    assert isinstance(compile_id, str) and compile_id
    best = db.best_compile("extraction")
    assert best is not None
    assert best["configuration"] == config
    assert best["optimizer"] == "BootstrapFewShot"
    assert best["artifact_path"] == "optimized_program.json"
    assert best["baseline_score"] == pytest.approx(0.6)
    assert best["compiled_score"] == pytest.approx(0.9)
    assert best["dataset_version"] == "extraction_v1"
    assert "created_at" in best
    db.close()


def test_best_compile_orders_by_score(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    for score in (0.4, 0.9, 0.6):
        db.record_compile(
            app_name="qa",
            dataset_version="qa_v1",
            optimizer="BootstrapFewShot",
            configuration={},
            artifact_path=f"program_{score}.json",
            baseline_score=0.3,
            compiled_score=score,
        )

    best = db.best_compile("qa")
    assert best is not None
    assert best["compiled_score"] == pytest.approx(0.9)
    assert best["artifact_path"] == "program_0.9.json"

    assert db.best_compile("extraction") is None
    db.close()
