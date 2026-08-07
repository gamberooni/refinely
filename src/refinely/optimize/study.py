"""Optuna study creation and execution for app optimization."""

from collections.abc import Callable
from pathlib import Path

import optuna
from optuna.samplers import TPESampler


def run_study(
    app_name: str,
    objective: Callable,
    lineage_db_path: str | Path,
    n_trials: int = 15,
) -> optuna.study.Study:
    """Create or resume the app's study and run `n_trials` optimization trials.

    The study is persisted to a SQLite storage backend on the same file as the
    experiment lineage database, so trials survive across CLI invocations.
    """
    storage = f"sqlite:///{Path(lineage_db_path).resolve()}"
    study = optuna.create_study(
        study_name=f"refinely_{app_name}",
        direction="maximize",
        sampler=TPESampler(),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    return study
