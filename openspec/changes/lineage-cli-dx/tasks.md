## 1. Dependency

- [ ] 1.1 Add `rich` to `[project.dependencies]` in `pyproject.toml` and sync the lockfile (`uv sync --group dev`)

## 2. LineageDB read API

- [ ] 2.1 Add `list_runs(app_name: str, limit: int = 50) -> list[dict]` to `LineageDB` in `src/crucible/tracking/db.py`: returns runs ordered by `created_at` DESC (limit applied), each dict with run_id, app_name, dataset_version, configuration (parsed), optuna_trial_number, aggregate_score, created_at, and `metric_results` (dict pivoted from `metric_results` table, Python-side join of two selects)
- [ ] 2.2 Add tests in `tests/test_tracking.py`: `list_runs` returns newest-first with metric values joined; respects limit; returns empty list for unknown app; parses configuration JSON

## 3. Reporting module

- [ ] 3.1 Create `src/crucible/reporting/__init__.py`
- [ ] 3.2 Implement rich table renderers in `src/crucible/reporting/render.py`: runs table (run id truncated, created_at, score, one column per metric union, trial number), cases table (case id, score, truncated input/expected/output), compare table (metric values + signed deltas vs. baseline, `(unchanged)` for zero deltas), best-run and best-compile summary panels
- [ ] 3.3 Implement `src/crucible/reporting/export.py`: CSV writer (stdlib `csv`, utf-8; columns run_id, created_at, aggregate_score, optuna_trial_number, then union of metric names, blank when absent) and JSON writer (stdlib `json`, `indent=2`, list of run objects); both take `list_runs` output and a target path and write the file

## 4. CLI commands

- [ ] 4.1 Add `show` command to `src/crucible/cli.py`: `crucible show <app> [--run <run_id>] [--limit N]` — runs table (newest first, default limit 50) + best run/best compile panels; with `--run`, renders per-case table via `case_results_for_run`; unknown run id raises `ClickException`
- [ ] 4.2 Add `compare` command: `crucible compare <app> [--baseline <run_id>]` — chronological table with per-metric deltas vs. previous run by default or vs. `--baseline`; first run (or baseline run) marked with no deltas; unknown baseline raises `ClickException`
- [ ] 4.3 Add `export` command: `crucible export <app> [--format csv|json] [--output FILE]` — `--format` is a click Choice defaulting to `csv`; `--output` defaults to `<app>_runs.csv`/`<app>_runs.json` in cwd; always writes the file and echoes the path (also when the app has no runs)
- [ ] 4.4 Wrap `evaluate`, `optimize`, and `compile` output in rich panels in `src/crucible/cli.py`, keeping all printed data values and labels unchanged
- [ ] 4.5 Add CLI tests in `tests/test_cli.py`: `show` with runs (asserts table rows and best-run summary), `show` with no runs, `show --run` drill-down (worst first), `show --run` unknown id errors, `compare` previous-run deltas, `compare --baseline` deltas, `compare` unknown baseline errors, `export` csv (file written, path echoed, columns correct), `export --format json`, `export` with no runs writes empty file, `export --format yaml` rejected
- [ ] 4.6 Verify existing suite passes: `uv run pytest tests/ -q` (119 tests + new ones, no network)

## 5. Spec sync

- [ ] 5.1 Sync delta specs to main specs after implementation: `openspec sync` for `cli` and `experiment-lineage-tracking` capabilities (add the new requirements from `specs/cli/spec.md`, `specs/experiment-lineage-tracking/spec.md`, and `specs/lineage-cli-read-back/spec.md` — the latter becomes a new `lineage-cli-read-back` capability spec)
- [ ] 5.2 Update `docs/overview.md` observability note to reflect that lineage read-back is now available via `crucible show/compare/export` (terminal + file export, no web UI)
