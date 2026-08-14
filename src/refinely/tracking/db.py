from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from sqlalchemy import (
    Column,
    Connection,
    Float,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    func,
    inspect,
    select,
    text,
)

from refinely.eval.runner import CaseResult
from refinely.tracking.models import CaseRecord, CompileRecord, EvaluationRun

_metadata = MetaData()


def _normalize_tags(tags: list[str] | None) -> str | None:
    """Normalize a tag list to a comma-separated string (or None).

    Splits each entry on commas, strips whitespace, drops empties, and dedupes
    preserving first-seen order.
    """
    if not tags:
        return None
    seen: list[str] = []
    for entry in tags:
        for part in entry.split(","):
            tag = part.strip()
            if tag and tag not in seen:
                seen.append(tag)
    return ",".join(seen) if seen else None


evaluation_runs_table = Table(
    "evaluation_runs",
    _metadata,
    Column("run_id", Text, primary_key=True),
    Column("app_name", Text, nullable=False),
    Column("dataset_version", Text, nullable=False),
    Column("configuration", Text, nullable=False),
    Column("model_name", Text),
    Column("judge_model", Text),
    Column("judge_prompt_version", Text),
    Column("tags", Text),
    Column("optuna_trial_number", Integer),
    Column("aggregate_score", Float, nullable=False),
    Column("created_at", Text, nullable=False),
)

# run_id FKs on the child tables were intentionally omitted: SQLite does not
# enforce foreign keys unless PRAGMA foreign_keys=ON is set, which it never is.
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
    Column("metric_scores", Text),
    Column("error", Text),
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
    Column("baseline_std", Float),
    Column("compiled_std", Float),
    Column("verdict", Text),
    Column("created_at", Text, nullable=False),
)

Index("idx_evaluation_runs_app", evaluation_runs_table.c.app_name, text("aggregate_score DESC"))

optimize_gates_table = Table(
    "optimize_gates",
    _metadata,
    Column("gate_id", Text, primary_key=True),
    Column("app_name", Text, nullable=False),
    Column("trial_number", Integer),
    Column("baseline_mean", Float),
    Column("baseline_std", Float),
    Column("candidate_mean", Float),
    Column("candidate_std", Float),
    Column("n_repeats", Integer),
    Column("verdict", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)
Index("idx_case_results_run", case_results_table.c.run_id, text("score ASC"))
Index("idx_dspy_compiles_app", dspy_compiles_table.c.app_name, text("compiled_score DESC"))


class LineageDB:
    """SQLite-backed experiment lineage tracking.

    Shares one database file with Optuna's internal trial storage; custom
    tables use distinct names so the two coexist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = path
        self._engine = create_engine(f"sqlite:///{path!s}")

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
        database file that already contains Optuna's internal tables.

        Columns added after the initial release are backfilled on existing
        databases with a defensive ``ALTER TABLE`` (``create_all`` only
        creates missing tables and never alters existing ones).
        """
        _metadata.create_all(self._engine)
        self._backfill_columns()

    def _backfill_columns(self) -> None:
        """Add columns introduced after the first schema release to existing tables."""
        inspector = inspect(self._engine)
        case_columns = {c["name"] for c in inspector.get_columns("case_results")}
        if "metric_scores" not in case_columns:
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE case_results ADD COLUMN metric_scores TEXT"))
        if "error" not in case_columns:
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE case_results ADD COLUMN error TEXT"))
        run_columns = {c["name"] for c in inspector.get_columns("evaluation_runs")}
        if "model_name" not in run_columns:
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE evaluation_runs ADD COLUMN model_name TEXT"))
        if "tags" not in run_columns:
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE evaluation_runs ADD COLUMN tags TEXT"))
        if "judge_model" not in run_columns:
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE evaluation_runs ADD COLUMN judge_model TEXT"))
        if "judge_prompt_version" not in run_columns:
            with self._engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE evaluation_runs ADD COLUMN judge_prompt_version TEXT")
                )
        compile_columns = {c["name"] for c in inspector.get_columns("dspy_compiles")}
        if "baseline_std" not in compile_columns:
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE dspy_compiles ADD COLUMN baseline_std FLOAT"))
        if "compiled_std" not in compile_columns:
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE dspy_compiles ADD COLUMN compiled_std FLOAT"))
        if "verdict" not in compile_columns:
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE dspy_compiles ADD COLUMN verdict TEXT"))

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
        model_name: str | None = None,
        tags: list[str] | None = None,
        judge_model: str | None = None,
        judge_prompt_version: str | None = None,
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
                    model_name=model_name,
                    tags=_normalize_tags(tags),
                    judge_model=judge_model,
                    judge_prompt_version=judge_prompt_version,
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
                            "metric_scores": json.dumps(c.scores, sort_keys=True),
                            "error": c.error,
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
        return self._row_to_run(row) if row else None

    def case_results_for_run(self, run_id: str) -> list[CaseRecord]:
        """Fetch a run's case results ordered by score ascending (worst first)."""
        stmt = (
            select(
                case_results_table.c.case_id,
                case_results_table.c.input,
                case_results_table.c.output,
                case_results_table.c.expected,
                case_results_table.c.score,
                case_results_table.c.metric_scores,
                case_results_table.c.error,
            )
            .where(case_results_table.c.run_id == run_id)
            .order_by(case_results_table.c.score.asc())
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [
            CaseRecord(
                case_id=r["case_id"],
                input=json.loads(r["input"]),
                output=json.loads(r["output"]) if r["output"] is not None else None,
                expected=json.loads(r["expected"]),
                score=r["score"],
                metric_scores=json.loads(r["metric_scores"])
                if r["metric_scores"] is not None
                else {},
                error=r["error"],
            )
            for r in rows
        ]

    def run_exists(self, run_id: str) -> bool:
        """Return whether a run with the given id is recorded."""
        stmt = (
            select(evaluation_runs_table.c.run_id)
            .where(evaluation_runs_table.c.run_id == run_id)
            .limit(1)
        )
        with self._engine.connect() as conn:
            return conn.execute(stmt).first() is not None

    def find_runs_by_prefix(self, app_name: str, prefix: str) -> list[str]:
        """Return an app's run ids starting with the given prefix, newest first.

        Used to resolve abbreviated run ids (e.g. the 8-char prefix shown by
        ``show``) to a full run id.
        """
        stmt = (
            select(evaluation_runs_table.c.run_id)
            .where(evaluation_runs_table.c.app_name == app_name)
            .where(evaluation_runs_table.c.run_id.like(prefix + "%"))
            .order_by(evaluation_runs_table.c.created_at.desc())
        )
        with self._engine.connect() as conn:
            return [row[0] for row in conn.execute(stmt)]

    def get_run(self, run_id: str) -> EvaluationRun | None:
        """Fetch a single run with per-metric values joined, or None."""
        stmt = (
            select(evaluation_runs_table).where(evaluation_runs_table.c.run_id == run_id).limit(1)
        )
        with self._engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
            if row is None:
                return None
            run = self._row_to_run(row)
            run.metric_results = self._metrics_by_run_id(conn, [run_id]).get(run_id, {})
        return run

    def list_runs(
        self,
        app_name: str,
        limit: int | None = 50,
        offset: int = 0,
        model_name: str | None = None,
        tag: str | None = None,
    ) -> list[EvaluationRun]:
        """Fetch an app's runs newest first with per-metric values joined.

        Metrics are pivoted into a ``metric_results`` dict per run via a
        Python-side join of two selects, since metric sets vary across apps.
        Optionally restrict to runs recorded with a given model name or tag.
        A tag filter is applied to the full matching set before any LIMIT,
        so filtered read-back is never silently truncated.
        """
        runs_stmt = select(evaluation_runs_table).where(
            evaluation_runs_table.c.app_name == app_name
        )
        if model_name is not None:
            runs_stmt = runs_stmt.where(evaluation_runs_table.c.model_name == model_name)
        runs_stmt = runs_stmt.order_by(evaluation_runs_table.c.created_at.desc())
        if tag is None and limit is not None:
            runs_stmt = runs_stmt.limit(limit).offset(offset)
        with self._engine.connect() as conn:
            run_rows = conn.execute(runs_stmt).mappings().all()
            if not run_rows:
                return []
            metrics_by_run = self._metrics_by_run_id(conn, [r["run_id"] for r in run_rows])

        runs = []
        for row in run_rows:
            run = self._row_to_run(row)
            run.metric_results = metrics_by_run.get(row["run_id"], {})
            runs.append(run)

        if tag is not None:
            runs = [
                r
                for r in runs
                if r.tags is not None and tag in [t.strip() for t in r.tags.split(",")]
            ]
            if limit is not None:
                runs = runs[offset : offset + limit]
        return runs

    def _metrics_by_run_id(
        self, conn: Connection, run_ids: list[str]
    ) -> dict[str, dict[str, float]]:
        metric_stmt = select(
            metric_results_table.c.run_id,
            metric_results_table.c.metric_name,
            metric_results_table.c.value,
        ).where(metric_results_table.c.run_id.in_(run_ids))
        metric_rows = conn.execute(metric_stmt).mappings().all()
        metrics_by_run: dict[str, dict[str, float]] = {}
        for m in metric_rows:
            metrics_by_run.setdefault(m["run_id"], {})[m["metric_name"]] = m["value"]
        return metrics_by_run

    def count_runs(self, app_name: str | None = None, model_name: str | None = None) -> int:
        stmt = select(func.count()).select_from(evaluation_runs_table)
        if app_name is not None:
            stmt = stmt.where(evaluation_runs_table.c.app_name == app_name)
        if model_name is not None:
            stmt = stmt.where(evaluation_runs_table.c.model_name == model_name)
        with self._engine.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def record_gate(
        self,
        app_name: str,
        trial_number: int | None,
        baseline_mean: float,
        baseline_std: float,
        candidate_mean: float,
        candidate_std: float,
        n_repeats: int,
        verdict: str,
    ) -> str:
        """Insert one optimize significance-gate row. Returns the generated gate_id."""
        gate_id = uuid.uuid4().hex
        created_at = datetime.now(UTC).isoformat()
        with self._engine.begin() as conn:
            conn.execute(
                optimize_gates_table.insert().values(
                    gate_id=gate_id,
                    app_name=app_name,
                    trial_number=trial_number,
                    baseline_mean=baseline_mean,
                    baseline_std=baseline_std,
                    candidate_mean=candidate_mean,
                    candidate_std=candidate_std,
                    n_repeats=n_repeats,
                    verdict=verdict,
                    created_at=created_at,
                )
            )
        return gate_id

    def record_compile(
        self,
        app_name: str,
        dataset_version: str,
        optimizer: str,
        configuration: dict[str, Any],
        artifact_path: str,
        baseline_score: float,
        compiled_score: float,
        baseline_std: float | None = None,
        compiled_std: float | None = None,
        verdict: str | None = None,
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
                    baseline_std=baseline_std,
                    compiled_std=compiled_std,
                    verdict=verdict,
                    created_at=created_at,
                )
            )
        return compile_id

    def best_compile(self, app_name: str) -> CompileRecord | None:
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
        return CompileRecord(**data)

    def _row_to_run(
        self, row: Mapping[str, Any], metric_results: dict[str, float] | None = None
    ) -> EvaluationRun:
        data = dict(row)
        data["configuration"] = json.loads(data["configuration"])
        if metric_results is not None:
            data["metric_results"] = metric_results
        return EvaluationRun(**data)
