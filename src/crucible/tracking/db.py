from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Self

from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    func,
    select,
    text,
)

from crucible.eval.runner import CaseResult

_metadata = MetaData()

evaluation_runs_table = Table(
    "evaluation_runs",
    _metadata,
    Column("run_id", Text, primary_key=True),
    Column("app_name", Text, nullable=False),
    Column("dataset_version", Text, nullable=False),
    Column("configuration", Text, nullable=False),
    Column("optuna_trial_number", Integer),
    Column("aggregate_score", Float, nullable=False),
    Column("created_at", Text, nullable=False),
)

metric_results_table = Table(
    "metric_results",
    _metadata,
    Column("run_id", Text, nullable=False),
    Column("metric_name", Text, nullable=False),
    Column("value", Float, nullable=False),
)

case_results_table = Table(
    "case_results",
    _metadata,
    Column("run_id", Text, nullable=False),
    Column("case_id", Text, nullable=False),
    Column("input", Text, nullable=False),
    Column("output", Text),
    Column("expected", Text, nullable=False),
    Column("score", Float, nullable=False),
)

dspy_compiles_table = Table(
    "dspy_compiles",
    _metadata,
    Column("compile_id", Text, primary_key=True),
    Column("app_name", Text, nullable=False),
    Column("dataset_version", Text, nullable=False),
    Column("optimizer", Text, nullable=False),
    Column("configuration", Text, nullable=False),
    Column("artifact_path", Text, nullable=False),
    Column("baseline_score", Float, nullable=False),
    Column("compiled_score", Float, nullable=False),
    Column("created_at", Text, nullable=False),
)

Index("idx_evaluation_runs_app", evaluation_runs_table.c.app_name, text("aggregate_score DESC"))
Index("idx_case_results_run", case_results_table.c.run_id, text("score ASC"))
Index("idx_dspy_compiles_app", dspy_compiles_table.c.app_name, text("compiled_score DESC"))


class LineageDB:
    """SQLite-backed experiment lineage tracking.

    Shares one database file with Optuna's internal trial storage; custom
    tables use distinct names so the two coexist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = path
        self._engine = create_engine(f"sqlite:///{str(path)}")

    @property
    def path(self) -> str:
        return str(self._path)

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self) -> Self:
        self.init_schema()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def init_schema(self) -> None:
        """Create the lineage tables if missing. Idempotent and safe on a
        database file that already contains Optuna's internal tables."""
        _metadata.create_all(self._engine)

    def record_run(
        self,
        app_name: str,
        dataset_version: str,
        configuration: dict[str, Any],
        aggregate_score: float,
        metric_results: dict[str, float],
        case_results: list[CaseResult],
        weights: dict[str, float],
        optuna_trial_number: int | None = None,
    ) -> str:
        """Insert one evaluation run plus its metric and case rows. Returns the generated run_id."""
        run_id = uuid.uuid4().hex
        created_at = datetime.now(UTC).isoformat()

        with self._engine.begin() as conn:
            conn.execute(
                evaluation_runs_table.insert().values(
                    run_id=run_id,
                    app_name=app_name,
                    dataset_version=dataset_version,
                    configuration=json.dumps(configuration, sort_keys=True),
                    optuna_trial_number=optuna_trial_number,
                    aggregate_score=aggregate_score,
                    created_at=created_at,
                )
            )
            if metric_results:
                conn.execute(
                    metric_results_table.insert(),
                    [
                        {"run_id": run_id, "metric_name": name, "value": value}
                        for name, value in metric_results.items()
                    ],
                )
            if case_results:
                conn.execute(
                    case_results_table.insert(),
                    [
                        {
                            "run_id": run_id,
                            "case_id": c.case_id,
                            "input": json.dumps(c.input, sort_keys=True),
                            "output": json.dumps(c.output, sort_keys=True)
                            if c.output is not None
                            else None,
                            "expected": json.dumps(c.expected, sort_keys=True),
                            "score": sum(
                                weights.get(name, 0.0) * value for name, value in c.scores.items()
                            ),
                        }
                        for c in case_results
                    ],
                )
        return run_id

    def best_run(self, app_name: str) -> dict[str, Any] | None:
        """Fetch the highest-scoring run for an app, config parsed back to a dict."""
        stmt = (
            select(evaluation_runs_table)
            .where(evaluation_runs_table.c.app_name == app_name)
            .order_by(
                evaluation_runs_table.c.aggregate_score.desc(),
                evaluation_runs_table.c.created_at.asc(),
            )
            .limit(1)
        )
        with self._engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return self._run_row_to_dict(row) if row else None

    def case_results_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch a run's case results ordered by score ascending (worst first)."""
        stmt = (
            select(
                case_results_table.c.case_id,
                case_results_table.c.input,
                case_results_table.c.output,
                case_results_table.c.expected,
                case_results_table.c.score,
            )
            .where(case_results_table.c.run_id == run_id)
            .order_by(case_results_table.c.score.asc())
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [
            {
                "case_id": r["case_id"],
                "input": json.loads(r["input"]),
                "output": json.loads(r["output"]) if r["output"] is not None else None,
                "expected": json.loads(r["expected"]),
                "score": r["score"],
            }
            for r in rows
        ]

    def count_runs(self, app_name: str | None = None) -> int:
        stmt = select(func.count()).select_from(evaluation_runs_table)
        if app_name is not None:
            stmt = stmt.where(evaluation_runs_table.c.app_name == app_name)
        with self._engine.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def record_compile(
        self,
        app_name: str,
        dataset_version: str,
        optimizer: str,
        configuration: dict[str, Any],
        artifact_path: str,
        baseline_score: float,
        compiled_score: float,
    ) -> str:
        """Insert one DSPy compile row. Returns the generated compile_id."""
        compile_id = uuid.uuid4().hex
        created_at = datetime.now(UTC).isoformat()
        with self._engine.begin() as conn:
            conn.execute(
                dspy_compiles_table.insert().values(
                    compile_id=compile_id,
                    app_name=app_name,
                    dataset_version=dataset_version,
                    optimizer=optimizer,
                    configuration=json.dumps(configuration, sort_keys=True),
                    artifact_path=artifact_path,
                    baseline_score=baseline_score,
                    compiled_score=compiled_score,
                    created_at=created_at,
                )
            )
        return compile_id

    def best_compile(self, app_name: str) -> dict[str, Any] | None:
        """Fetch the highest-scoring compile for an app, config parsed back to a dict."""
        stmt = (
            select(dspy_compiles_table)
            .where(dspy_compiles_table.c.app_name == app_name)
            .order_by(
                dspy_compiles_table.c.compiled_score.desc(), dspy_compiles_table.c.created_at.asc()
            )
            .limit(1)
        )
        with self._engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        data = dict(row)
        data["configuration"] = json.loads(data["configuration"])
        return data

    def _run_row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        data = dict(row)
        data["configuration"] = json.loads(data["configuration"])
        return data
