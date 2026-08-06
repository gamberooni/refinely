# Design: actionable-results

## Context

Lineage read-back (change `lineage-cli-dx`) gives the user `show`, `compare`, and `export`, but the data they operate on is coarse: `compare` shows aggregate per-metric deltas only, per-case errors are deliberately kept in memory (`case_results` has no error column), and runs accumulate with no way to mark or filter meaningful ones. The evaluation runner already captures `CaseResult.error` (str|None) per case during a run — it is simply dropped before persistence.

`evaluation_runs.configuration` already stores the merged config JSON per run, so config deltas are derivable without new storage. The schema migration pattern already exists: `init_schema` → `_backfill_columns()` performs guarded `ALTER TABLE ... ADD COLUMN` based on an SQLAlchemy inspector check (introduced when `metric_scores` was added).

## Goals / Non-Goals

**Goals:**
- Let users tag runs at creation and filter all read-back by tag.
- Let users see *which* config keys changed between runs (`compare --diff-config`).
- Let users see *which cases* broke/fixed between two runs (`compare --cases`).
- Persist per-case errors so failures survive the run and are visible in `show --run`.

**Non-Goals:**
- Retroactive re-tagging of existing runs (no `tag`/`untag` commands).
- Automatic regression alerting or thresholds.
- Pairing cases by anything other than index/position (e.g. no cross-dataset content matching).
- Per-case error aggregation metrics or error-rate columns on `evaluation_runs`.

## Decisions

### D1: Tags stored as a normalized comma-separated TEXT column
`evaluation_runs` gains `tags TEXT` holding a comma-separated list (e.g. `"candidate,prod"`), normalized at write time: split on commas, strip, drop empties, dedupe preserving order. NULL means "no tags".

*Why over JSON list:* the read path (`list_runs`) already does Python-side per-row post-processing (metric pivot); a flat string avoids parse ceremony for a plain list, stays human-readable in raw `sqlite3`, and needs no nested structure. Filtering happens in Python (`tag in tags.split(",")`), consistent with the existing pivot.

### D2: `error` column via the existing backfill pattern
`case_results` gains `error TEXT NULL`. `_backfill_columns` gets two new guarded entries: `ALTER TABLE evaluation_runs ADD COLUMN tags TEXT` and `ALTER TABLE case_results ADD COLUMN error TEXT`. Both guarded by the inspector's column check, preserving rows. This overturns the deliberate "errors are memory-only" design — flagged as breaking in the proposal; the migration is additive and NULL-for-clean-cases.

### D3: `record_run` threads tags and per-case errors
`LineageDB.record_run` gains `tags: list[str] | None = None` and records each `CaseResult`'s existing `error` field into the `case_results.error` column. The evaluation loop is unchanged — the runner already captures errors in `CaseResult.error`; the record path just stops dropping them. `EvaluationRun` gains `tags: str | None = None`; `case_results_for_run` returns the error with each `CaseRecord`.

### D4: Tag filter lives in `list_runs`
`list_runs(app_name, limit, offset, tag=None)` filters after the existing per-row assembly (`tag in row.tags.split(",")` when tags present). All three read-back commands (`show`, `compare`, `export`) call the same filtered path, so filter behavior is uniform. Empty match → caller prints "no runs found matching the tag".

### D5: `compare --diff-config` = key-level delta vs baseline
For each compared run, compute `added` (keys in run config not in baseline), `removed`, and `changed` (keys with differing values) against the baseline run's `configuration`. Rendered as a section next to the metric deltas; equal configs render an explicit no-change marker. Reuses existing config JSON already stored per run — no new storage.

### D6: `compare --cases` pairs baseline + most recent run
`--cases` produces exactly one paired table: the baseline run (explicit `--baseline`, or the predecessor of the newest run when omitted) vs the newest run. Cases pair by index; the table shows per-case score before/after + Δ. Summary line "N broke / M fixed / K unchanged" (broke = score decreased, fixed = increased, unchanged = equal). When the two runs' `dataset_version` differ, print a warning that index pairing may be meaningless. Require ≥2 runs (or `--baseline` + any newer run) else a "needs two runs" message. Runs with differing case counts pair up to the shorter list.

### D7: Rendering additions in `reporting/render.py`
- `cases_table` gains an `error` column (blank for clean cases); `show --run` appends an "N cases errored" line when N > 0.
- `compare_table` accepts an optional config-delta section renderer.
- New `case_pair_table` for the `--cases` paired output.
No changes to existing columns/ordering of the runs table.

## Risks / Trade-offs

- **Breaking the memory-only error design** → Deliberate and additive; existing rows get NULL errors; `show --run` renders blank for clean cases, so prior workflows still render.
- **`record_run` has ~14 call sites** → Default `tags=None` keeps every existing caller working; only the CLI threading (evaluate/optimize) and the runner record path change.
- **`--cases` pairing by index is fragile across dataset edits** → Dataset versions are recorded per run; the explicit `dataset_version` warning surfaces the risk, and pairing-by-content is a stated non-goal.
- **Tags normalization rules are invisible to users** → Commands echo back effective tags in run creation output; normalization is deterministic (strip/dedupe/drop-empty).

## Migration Plan

Schema upgrade happens automatically on next `LineageDB` open via `init_schema` → `_backfill_columns` (guarded ADD COLUMN for `tags` and `error`). No data rewrite; rollback is not applicable for SQLite additive columns (a downgrade would drop the columns, discarding tags/errors).

## Open Questions

- None blocking. (Candidate for later: rendering tags in the `show` runs table — currently tags surface only via filtering, not as a visible column.)
