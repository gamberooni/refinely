# Proposal: actionable-results

## Why

Lineage read-back answers "what scored what" but not "what changed and why": per-case errors exist only in memory (deliberately unpersisted), `compare` shows aggregate deltas but never *which cases* broke, configs are visible only by eyeballing JSON, and runs accumulate with no way to mark or filter meaningful ones. The tool can't yet tell you "this config fixed 12 cases and broke these 3".

## What Changes

- **Run tags at creation**: `evaluate`/`optimize` accept `--tags <a,b>` (e.g. `candidate,prod`); tags are persisted on `evaluation_runs` and `show`/`compare`/`export` gain a `--tag` filter. Retroactive re-tagging is out of scope.
- **Config diff in compare**: `compare` gains `--diff-config` — a config delta section alongside the per-metric deltas (configs are already stored per run).
- **Per-case regression drilldown**: `compare` gains `--cases` — a paired per-case delta table (before/after per metric, direction broke/fixed/unchanged) plus a "N broke / M fixed / K unchanged" summary. When the two runs' `dataset_version` differ, the CLI warns that index pairing may be meaningless.
- **Persist per-case errors** (**BREAKING** to the "errors are memory-only" design): add a nullable `error` TEXT column to `case_results`; `show --run` renders an error column and the run read-back reports how many cases errored.

## Capabilities

### New Capabilities
- `run-tags`: associating a comma-separated tag list with a run at creation, persisting it on `evaluation_runs`, and filtering run read-back (show/compare/export) by tag.

### Modified Capabilities
- `experiment-lineage-tracking`: `evaluation_runs` gains a `tags` column; `case_results` gains a nullable `error` column; schema init upgrades existing databases by adding the columns without losing rows; run recording stores tags and per-case errors.
- `lineage-cli-read-back`: `show --run` renders per-case error column + errored count; `compare` gains `--diff-config` (config delta) and `--cases` (paired per-case regression table with broke/fixed/unchanged summary and dataset_version warning); `show`/`compare`/`export` accept a `--tag` filter.
- `cli`: `evaluate` and `optimize` accept a `--tags` flag at run creation.

## Impact

- `src/crucible/tracking/db.py` — `tags` + `error` columns, backfill entries, `record_run` signature (tags, per-case error), `list_runs` tag filter, `case_results_for_run` error field.
- `src/crucible/reporting/render.py` — error column in `cases_table`; `compare_table` gains optional config-delta section; new paired per-case delta table for `--cases`.
- `src/crucible/cli.py` — `--tags` on evaluate/optimize; `--tag` filter on show/compare/export; `--diff-config`/`--cases` flags on compare.
- `src/crucible/eval/runner.py` — `CaseResult` gains an `error` field passed through from the existing in-memory per-case error capture (no behavior change to the loop itself).
- No new dependencies.
