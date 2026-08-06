from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from crucible.eval.runner import CaseResult


class LineageDB:
    """SQLite-backed experiment lineage tracking.

    Shares one database file with Optuna's internal trial storage; custom
    tables use distinct names so the two coexist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row

    @property
    def path(self) -> str:
        return str(self._path)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        self.init_schema()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def init_schema(self) -> None:
        """Create the lineage tables if missing. Idempotent and safe on a
        database file that already contains Optuna's internal tables."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id             TEXT PRIMARY KEY,
                app_name           TEXT NOT NULL,
                dataset_version    TEXT NOT NULL,
                configuration      TEXT NOT NULL,
                optuna_trial_number INTEGER,
                aggregate_score    REAL NOT NULL,
                created_at         TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metric_results (
                run_id      TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value       REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES evaluation_runs (run_id)
            );

            CREATE TABLE IF NOT EXISTS case_results (
                run_id   TEXT NOT NULL,
                case_id  TEXT NOT NULL,
                input    TEXT NOT NULL,
                output   TEXT,
                expected TEXT NOT NULL,
                score    REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES evaluation_runs (run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_evaluation_runs_app
                ON evaluation_runs (app_name, aggregate_score DESC);
            CREATE INDEX IF NOT EXISTS idx_case_results_run
                ON case_results (run_id, score ASC);

            CREATE TABLE IF NOT EXISTS dspy_compiles (
                compile_id      TEXT PRIMARY KEY,
                app_name        TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                optimizer       TEXT NOT NULL,
                configuration   TEXT NOT NULL,
                artifact_path   TEXT NOT NULL,
                baseline_score  REAL NOT NULL,
                compiled_score  REAL NOT NULL,
                created_at      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_dspy_compiles_app
                ON dspy_compiles (app_name, compiled_score DESC);
            """
        )
        self._conn.commit()

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

        self._conn.execute(
            "INSERT INTO evaluation_runs "
            "(run_id, app_name, dataset_version, configuration, optuna_trial_number, aggregate_score, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                app_name,
                dataset_version,
                json.dumps(configuration, sort_keys=True),
                optuna_trial_number,
                aggregate_score,
                created_at,
            ),
        )
        self._conn.executemany(
            "INSERT INTO metric_results (run_id, metric_name, value) VALUES (?, ?, ?)",
            [(run_id, name, value) for name, value in metric_results.items()],
        )
        self._conn.executemany(
            "INSERT INTO case_results (run_id, case_id, input, output, expected, score) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    c.case_id,
                    json.dumps(c.input, sort_keys=True),
                    json.dumps(c.output, sort_keys=True) if c.output is not None else None,
                    json.dumps(c.expected, sort_keys=True),
                    sum(weights.get(name, 0.0) * value for name, value in c.scores.items()),
                )
                for c in case_results
            ],
        )
        self._conn.commit()
        return run_id

    def best_run(self, app_name: str) -> dict[str, Any] | None:
        """Fetch the highest-scoring run for an app, config parsed back to a dict."""
        row = self._conn.execute(
            "SELECT * FROM evaluation_runs "
            "WHERE app_name = ? ORDER BY aggregate_score DESC, created_at ASC LIMIT 1",
            (app_name,),
        ).fetchone()
        return self._run_row_to_dict(row) if row else None

    def case_results_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch a run's case results ordered by score ascending (worst first)."""
        rows = self._conn.execute(
            "SELECT case_id, input, output, expected, score FROM case_results "
            "WHERE run_id = ? ORDER BY score ASC",
            (run_id,),
        ).fetchall()
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
        if app_name is None:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM evaluation_runs").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM evaluation_runs WHERE app_name = ?",
                (app_name,),
            ).fetchone()
        return int(row["n"])

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
        self._conn.execute(
            "INSERT INTO dspy_compiles "
            "(compile_id, app_name, dataset_version, optimizer, configuration, artifact_path, "
            " baseline_score, compiled_score, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                compile_id,
                app_name,
                dataset_version,
                optimizer,
                json.dumps(configuration, sort_keys=True),
                artifact_path,
                baseline_score,
                compiled_score,
                created_at,
            ),
        )
        self._conn.commit()
        return compile_id

    def best_compile(self, app_name: str) -> dict[str, Any] | None:
        """Fetch the highest-scoring compile for an app, config parsed back to a dict."""
        row = self._conn.execute(
            "SELECT * FROM dspy_compiles "
            "WHERE app_name = ? ORDER BY compiled_score DESC, created_at ASC LIMIT 1",
            (app_name,),
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["configuration"] = json.loads(data["configuration"])
        return data

    def _run_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["configuration"] = json.loads(data["configuration"])
        return data
