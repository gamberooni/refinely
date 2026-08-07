## Why

All evaluation, optimization, and compile results are persisted to the lineage SQLite database, but the only way to read them back is raw `sqlite3` queries. Users cannot see run history, drill into low-scoring cases, compare runs, or export data for external analysis — the CLI offers no read-back surface at all.

## What Changes

- Add `rich` as a hard runtime dependency for terminal formatting.
- Add `list_runs(app_name, limit)` read method to `LineageDB` returning runs newest-first with metric values joined (pivoted per row).
- Add `refinely show <app>` — rich table of recent runs (newest first) plus best run and best compile summary.
- Add `refinely show <app> --run <run_id>` — per-case drill-down, worst cases first (uses existing `case_results_for_run`).
- Add `refinely compare <app>` — metric table comparing each run against the previous run (chronological), with optional `--baseline <run_id>` to compare against a specific run instead.
- Add `refinely export <app> [--format csv|json] [--output FILE]` — writes runs + metrics to a file; `--format` defaults to `csv`; `--output` defaults to `<app>_runs.csv` (or `.json`) in the current directory; always writes a file.
- Reformat `evaluate`, `optimize`, and `compile` command output using `rich` panels (cosmetic only; data content unchanged).
- Sync new requirements into `cli` and `experiment-lineage-tracking` specs.

## Capabilities

### New Capabilities

- `lineage-cli-read-back`: CLI commands (`show`, `compare`, `export`) that surface lineage database contents — run history, per-case drill-down, run comparison, and file export.

### Modified Capabilities

- `cli`: new subcommands (`show`, `compare`, `export`) added to the CLI surface; existing commands' output reformatted with rich panels.
- `experiment-lineage-tracking`: new `list_runs` read query returning runs newest-first with metrics joined.

## Impact

- **Code**: `src/refinely/cli.py` (3 new commands + output formatting), `src/refinely/tracking/db.py` (one new query method), `src/refinely/reporting/` (new module for rendering/export logic).
- **Dependencies**: `rich` added to `[project.dependencies]` (hard dependency).
- **Tests**: `tests/test_tracking.py` (list_runs coverage), `tests/test_cli.py` (new command coverage).
- **Docs**: `openspec/specs/cli/spec.md`, `openspec/specs/experiment-lineage-tracking/spec.md` updated.
- **No breaking changes** to existing commands' data output or the lineage schema.
