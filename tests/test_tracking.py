import sqlite3

import pytest
from sqlalchemy import text

from apps.extraction import EXTRACTION_WEIGHTS
from apps.qa import QA_WEIGHTS
from refinely.eval.runner import CaseResult
from refinely.tracking.db import LineageDB, evaluation_runs_table


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
    assert best.configuration == config
    assert best.aggregate_score == pytest.approx(0.85)
    assert best.optuna_trial_number == 7
    assert best.dataset_version == "extraction_v1"

    cases = db.case_results_for_run(run_id)
    assert len(cases) == 3
    by_id = {c.case_id: c for c in cases}
    assert by_id["c0"].input == {"text": "input 0"}
    assert by_id["c0"].output == {"field_value": "positive"}
    assert by_id["c0"].expected == {
        "field_name": "sentiment",
        "field_value": "positive",
    }
    assert by_id["c0"].score == pytest.approx(0.7 * 1.0 + 0.15 * 1.0 + 0.15 * 1.0)
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
    assert best.aggregate_score == pytest.approx(0.9)

    assert db.best_run("extraction") is None
    db.close()


def test_list_runs_returns_newest_first_with_metrics_joined(
    tmp_path: pytest.TempPathFactory,
) -> None:
    from datetime import UTC, datetime

    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    run_ids = []
    for i, score in enumerate((0.4, 0.9, 0.6)):
        run_id = db.record_run(
            app_name="extraction",
            dataset_version="extraction_v1",
            configuration={"temperature": 0.1 + i},
            aggregate_score=score,
            metric_results={"exact_match": score, "latency": 1.0},
            case_results=_case_results(1),
            weights=EXTRACTION_WEIGHTS,
            optuna_trial_number=i,
        )
        run_ids.append(run_id)
        with db._engine.begin() as conn:
            conn.execute(
                evaluation_runs_table.update()
                .where(evaluation_runs_table.c.run_id == run_id)
                .values(created_at=datetime(2026, 1, i + 1, tzinfo=UTC).isoformat())
            )

    runs = db.list_runs("extraction")

    assert [r.run_id for r in runs] == list(reversed(run_ids))
    assert runs[0].metric_results == {"exact_match": 0.6, "latency": 1.0}
    assert runs[0].configuration == {"temperature": 2.1}
    assert runs[0].optuna_trial_number == 2
    assert runs[0].aggregate_score == pytest.approx(0.6)
    assert runs[0].dataset_version == "extraction_v1"
    assert runs[0].created_at
    db.close()


def test_list_runs_respects_limit(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    for score in (0.4, 0.9, 0.6, 0.7, 0.5):
        db.record_run(
            app_name="qa",
            dataset_version="qa_v1",
            configuration={"temperature": 0.1},
            aggregate_score=score,
            metric_results={"fuzzy_match": score},
            case_results=_case_results(1),
            weights=QA_WEIGHTS,
        )

    runs = db.list_runs("qa", limit=2)

    assert len(runs) == 2
    assert [r.aggregate_score for r in runs] == [pytest.approx(0.5), pytest.approx(0.7)]
    db.close()


def test_list_runs_empty_for_unknown_app(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration={},
        aggregate_score=0.5,
        metric_results={"exact_match": 0.5},
        case_results=_case_results(1),
        weights=EXTRACTION_WEIGHTS,
    )

    assert db.list_runs("rag") == []
    db.close()


def test_list_runs_parses_configuration_json(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    config = {"temperature": 0.7, "system_prompt_variant": "strict"}
    db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration=config,
        aggregate_score=0.8,
        metric_results={"exact_match": 0.8},
        case_results=_case_results(1),
        weights=EXTRACTION_WEIGHTS,
    )

    runs = db.list_runs("extraction")

    assert runs[0].configuration == config
    db.close()


def test_run_exists_true_for_recorded_run(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    run_id = db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration={},
        aggregate_score=0.5,
        metric_results={"exact_match": 0.5},
        case_results=_case_results(1),
        weights=EXTRACTION_WEIGHTS,
    )

    assert db.run_exists(run_id) is True
    db.close()


def test_run_exists_false_for_unknown_run(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()

    assert db.run_exists("does-not-exist") is False
    db.close()


def test_list_runs_respects_offset(tmp_path: pytest.TempPathFactory) -> None:
    from datetime import UTC, datetime

    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    run_ids = []
    for i in range(5):
        run_id = db.record_run(
            app_name="extraction",
            dataset_version="extraction_v1",
            configuration={},
            aggregate_score=0.1 + i,
            metric_results={"exact_match": 0.1 + i},
            case_results=_case_results(1),
            weights=EXTRACTION_WEIGHTS,
        )
        run_ids.append(run_id)
        with db._engine.begin() as conn:
            conn.execute(
                evaluation_runs_table.update()
                .where(evaluation_runs_table.c.run_id == run_id)
                .values(created_at=datetime(2026, 1, i + 1, tzinfo=UTC).isoformat())
            )

    runs = db.list_runs("extraction", limit=2, offset=2)

    assert [r.run_id for r in runs] == [run_ids[2], run_ids[1]]
    db.close()


def test_get_run_returns_run_with_metrics(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    run_id = db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration={"temperature": 0.5},
        aggregate_score=0.7,
        metric_results={"exact_match": 0.7, "latency": 1.0},
        case_results=_case_results(1),
        weights=EXTRACTION_WEIGHTS,
    )

    run = db.get_run(run_id)

    assert run is not None
    assert run.run_id == run_id
    assert run.metric_results == {"exact_match": 0.7, "latency": 1.0}
    assert run.configuration == {"temperature": 0.5}
    assert db.get_run("does-not-exist") is None
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
    assert [c.case_id for c in result] == ["c2", "c0", "c1"]
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
    assert best.configuration == config
    assert best.optimizer == "BootstrapFewShot"
    assert best.artifact_path == "optimized_program.json"
    assert best.baseline_score == pytest.approx(0.6)
    assert best.compiled_score == pytest.approx(0.9)
    assert best.dataset_version == "extraction_v1"
    assert best.created_at
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
    assert best.compiled_score == pytest.approx(0.9)
    assert best.artifact_path == "program_0.9.json"

    assert db.best_compile("extraction") is None
    db.close()


def test_case_results_include_metric_scores(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    cases = _case_results(2)
    run_id = db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration={},
        aggregate_score=0.5,
        metric_results={"exact_match": 0.5, "latency": 1.0, "cost": 1.0},
        case_results=cases,
        weights=EXTRACTION_WEIGHTS,
    )

    result = db.case_results_for_run(run_id)

    by_id = {c.case_id: c for c in result}
    assert by_id["c0"].metric_scores == {"exact_match": 1.0, "latency": 1.0, "cost": 1.0}
    assert by_id["c1"].metric_scores == {"exact_match": 0.0, "latency": 1.0, "cost": 1.0}
    db.close()


def test_case_results_metric_scores_empty_when_missing(
    tmp_path: pytest.TempPathFactory,
) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    cases = _case_results(1)
    run_id = db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration={},
        aggregate_score=0.5,
        metric_results={},
        case_results=cases,
        weights=EXTRACTION_WEIGHTS,
    )
    with db._engine.begin() as conn:
        conn.execute(
            text("UPDATE case_results SET metric_scores = NULL WHERE run_id = :rid"),
            {"rid": run_id},
        )

    result = db.case_results_for_run(run_id)

    assert result[0].metric_scores == {}
    db.close()


def test_init_schema_backfills_metric_scores_column(
    tmp_path: pytest.TempPathFactory,
) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE case_results (run_id TEXT NOT NULL, case_id TEXT NOT NULL, "
        "input TEXT NOT NULL, output TEXT, expected TEXT NOT NULL, score FLOAT NOT NULL)"
    )
    conn.commit()
    conn.close()

    db = LineageDB(db_path)
    db.init_schema()
    db.close()

    conn = sqlite3.connect(db_path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(case_results)")}
    conn.close()
    assert "metric_scores" in cols


def test_find_runs_by_prefix_returns_matching_run_ids(
    tmp_path: pytest.TempPathFactory,
) -> None:
    from datetime import UTC, datetime

    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    run_ids = []
    for i, score in enumerate((0.4, 0.9, 0.6)):
        run_id = db.record_run(
            app_name="extraction",
            dataset_version="extraction_v1",
            configuration={},
            aggregate_score=score,
            metric_results={"exact_match": score},
            case_results=_case_results(1),
            weights=EXTRACTION_WEIGHTS,
        )
        run_ids.append(run_id)
        with db._engine.begin() as conn:
            conn.execute(
                evaluation_runs_table.update()
                .where(evaluation_runs_table.c.run_id == run_id)
                .values(created_at=datetime(2026, 1, i + 1, tzinfo=UTC).isoformat())
            )

    assert run_ids[0] in db.find_runs_by_prefix("extraction", run_ids[0][:8])
    assert db.find_runs_by_prefix("qa", run_ids[0][:8]) == []
    assert db.find_runs_by_prefix("extraction", "zzzzzzzz") == []
    db.close()


def _case_results_with_errors(n: int = 2) -> list[CaseResult]:
    return [
        CaseResult(
            case_id=f"c{i}",
            input={"text": f"input {i}"},
            output={"field_value": "positive"},
            expected={"field_name": "sentiment", "field_value": "positive"},
            scores={"exact_match": 1.0, "latency": 1.0, "cost": 1.0},
            error="boom" if i == 1 else None,
        )
        for i in range(n)
    ]


def test_init_schema_backfills_tags_and_error_columns(
    tmp_path: pytest.TempPathFactory,
) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE case_results (run_id TEXT NOT NULL, case_id TEXT NOT NULL, "
        "input TEXT NOT NULL, output TEXT, expected TEXT NOT NULL, score FLOAT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE evaluation_runs (run_id TEXT NOT NULL PRIMARY KEY, app_name TEXT NOT NULL, "
        "dataset_version TEXT NOT NULL, configuration TEXT NOT NULL, "
        "optuna_trial_number INTEGER, aggregate_score FLOAT NOT NULL, created_at TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

    db = LineageDB(db_path)
    db.init_schema()
    db.close()

    conn = sqlite3.connect(db_path)
    case_cols = {r[1] for r in conn.execute("PRAGMA table_info(case_results)")}
    run_cols = {r[1] for r in conn.execute("PRAGMA table_info(evaluation_runs)")}
    conn.close()
    assert {"metric_scores", "error"} <= case_cols
    assert {"model_name", "tags", "judge_model", "judge_prompt_version"} <= run_cols


def test_record_run_stores_normalized_tags_and_errors(
    tmp_path: pytest.TempPathFactory,
) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    run_id = db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration={},
        aggregate_score=0.85,
        metric_results={"exact_match": 1.0, "latency": 1.0, "cost": 1.0},
        case_results=_case_results_with_errors(),
        weights=EXTRACTION_WEIGHTS,
        tags=["candidate", "prod", "candidate"],
    )

    run = db.get_run(run_id)
    assert run is not None
    assert run.tags == "candidate,prod"

    cases = db.case_results_for_run(run_id)
    by_id = {c.case_id: c for c in cases}
    assert by_id["c0"].error is None
    assert by_id["c1"].error == "boom"


def test_record_run_tags_none_is_null(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    run_id = db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration={},
        aggregate_score=0.5,
        metric_results={},
        case_results=_case_results(1),
        weights=EXTRACTION_WEIGHTS,
    )

    run = db.get_run(run_id)
    assert run is not None
    assert run.tags is None


def test_list_runs_filters_by_tag(tmp_path: pytest.TempPathFactory) -> None:
    db = LineageDB(tmp_path / "lineage.db")
    db.init_schema()
    tagged = db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration={},
        aggregate_score=0.8,
        metric_results={"exact_match": 0.8},
        case_results=[],
        weights=EXTRACTION_WEIGHTS,
        tags=["candidate", "prod"],
    )
    db.record_run(
        app_name="extraction",
        dataset_version="extraction_v1",
        configuration={},
        aggregate_score=0.6,
        metric_results={"exact_match": 0.6},
        case_results=[],
        weights=EXTRACTION_WEIGHTS,
        tags=["prod"],
    )

    assert [r.run_id for r in db.list_runs("extraction", tag="candidate")] == [tagged]
    assert len(db.list_runs("extraction", tag="nope")) == 0
    assert len(db.list_runs("extraction")) == 2
