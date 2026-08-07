## Context

All experiment data already lands in the lineage SQLite DB (`lineage.db`): `evaluation_runs`, `metric_results`, `case_results`, and `dspy_compiles` tables, coexisting with Optuna's internal tables. The read API (`LineageDB.best_run`, `case_results_for_run`, `count_runs`, `best_compile`) covers single-run lookups only — there is no way to list history, compare runs, or export. The CLI (`evaluate`, `optimize`, `compile`) prints bare `click.echo` lines and has no read-back commands. `docs/overview.md` previously declared dashboards/UI out of scope; this change provides terminal and file read-back without adding a web UI.

## Goals / Non-Goals

**Goals:**
- One new read query on `LineageDB` (`list_runs`) that powers history, comparison, and export.
- Three new CLI commands: `show`, `compare`, `export`.
- Rich-formatted output (panels/tables) for all commands, including the existing three.
- All new behavior testable hermetically (no network, stub LLM, temp SQLite DBs).

**Non-Goals:**
- No web UI, dashboard, or charting library (sparklines/HTML reports are out of scope).
- No schema migration: `list_runs` reads existing tables only.
- No new lineage write paths, no new tables or columns.
- No change to the data content of `evaluate`/`optimize`/`compile` output (cosmetic reformatting only).
- No multi-app comparison (comparison is per app).

## Decisions

### D1: Add `rich` as a hard runtime dependency
`rich` (pure Python, no transitive weight) provides tables, panels, and text truncation needed by all four read/format features. Chosen over: `tabulate` (tables only, no panels), `textual` (interactive TUI — overkill for a prototype), manual ASCII formatting (reintroduces the alignment bugs rich solves), and zero-dep plain text (fails the "visualize more easily" goal). Rationale: the CLI is the primary user surface; `rich` is the standard terminal-formatting choice and keeps formatting code declarative. Alternative considered: optional dependency group — rejected because every CLI user benefits and the package is small.

### D2: Single new DB query `list_runs(app_name, limit=50)`
New method on `LineageDB` returning runs newest-first (`created_at` DESC) with metrics pivoted into a `metric_results` dict per run row. Implementation: two selects (runs + metric rows) joined in Python — a pivot in SQL would need per-metric CASE columns, which breaks when metric sets vary across runs (e.g., RAG has 6 metrics, extraction 3). Runs come back as dicts with parsed `configuration` and `metric_results`. `compare`'s chronological view and previous-run deltas are computed in the reporting layer from the same list (ascending by reversing the list), avoiding a second query path.

### D3: New `refinely/reporting/` module for rendering and export
Formatting/export logic lives in `src/refinely/reporting/` (`render.py` for rich tables/panels, `export.py` for CSV/JSON writers), keeping `cli.py` thin and matching the existing layering (CLI delegates to `eval/`, `optimize/`, `tracking/`). Each command: read via `LineageDB` → build data → render or export. CSV uses stdlib `csv` with utf-8 encoding; JSON uses stdlib `json` with `indent=2`. The same `list_runs` data feeds `show`, `compare`, and `export`, so deltas/formatting stay consistent.

### D4: Command shapes
- `refinely show <app>` — runs table (newest first, `--limit` default 50) + best-run and best-compile summary panels.
- `refinely show <app> --run <run_id>` — per-case table via existing `case_results_for_run`; unknown run id → `ClickException`.
- `refinely compare <app> [--baseline <run_id>]` — chronological table; each row's deltas vs. the immediately preceding run by default, vs. `--baseline` when given; first row (or baseline row) marked with no deltas; unknown baseline → `ClickException`.
- `refinely export <app> [--format csv|json] [--output FILE]` — `--format` choices `[csv, json]` (default `csv`) enforced by click `Choice`; `--output` defaults to `<app>_runs.csv`/`<app>_runs.json` in cwd; always writes a file and echoes the path. CSV columns: run_id, created_at, aggregate_score, optuna_trial_number, then one column per metric (union of metric names across runs, blank when absent).
- All commands read the DB through `Settings.lineage_db_path` (no new `--lineage-db` flag on read commands; the existing compile flag is out of scope for this change).

### D5: Rich panels for existing commands
`evaluate`, `optimize`, `compile` wrap their existing printed values in `rich` panels with the same labels and numbers (e.g., `aggregate_score: 0.8500` unchanged inside a Panel). Rationale: keeps any script that greps the values working while unifying look; the spec's "Rich-formatted terminal output" requirement pins this.

## Risks / Trade-offs

- **Scripts parsing existing output** → Data content and label lines are unchanged; only wrapping/decoration is added, so line-greppable values survive. Flagged as cosmetic in the proposal.
- **`rich` markup in captured output** → CLI tests using `CliRunner` capture the rendered ANSI/markup text; tests assert on data substrings, not exact markup, so formatter tweaks don't break the suite.
- **Metric sets vary across runs** → `list_runs` pivots in Python (D2) and `export`/`show` use the union of metric names, blank-filling missing values; no assumption that every run has the same metric set.
- **Large case tables / many runs** → `show`/`compare` render with truncation (`rich` Table `truncate`/`no_wrap`) and `--limit` on `show`; export is unbounded by design.
- **Rich is a new hard dependency** → Small pure-Python package with no build requirements; consistent with the project's terminal-first stance (D1).

## Migration Plan

No migration needed: schema is unchanged; new code is additive. Rollback = remove the new commands and `list_runs`; existing commands' data output is unchanged in content.

## Open Questions

None — all decision branches were resolved in planning (dependency, ordering, baseline semantics, export defaults, existing-command formatting).
