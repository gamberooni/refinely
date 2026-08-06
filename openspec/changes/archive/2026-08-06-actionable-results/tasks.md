# Tasks: actionable-results

## 1. Schema: tags and error columns

- [x] 1.1 Add `tags TEXT` to `evaluation_runs` metadata and `error TEXT` to `case_results` metadata in `src/crucible/tracking/db.py`
- [x] 1.2 Add two guarded `ALTER TABLE ... ADD COLUMN` backfill entries (tags on evaluation_runs, error on case_results) to `LineageDB._backfill_columns` following the existing `metric_scores` pattern (D2)
- [x] 1.3 Add `tags: str | None = None` to the `EvaluationRun` dataclass and `error` to the `CaseRecord` read model; `_row_to_run` picks both up
- [x] 1.4 Add `error` to `case_results_for_run` return values (None for clean cases)

## 2. Persist tags and per-case errors on write

- [x] 2.1 Add `tags: list[str] | None = None` param to `LineageDB.record_run`; normalize to a comma-separated string (split/strip/dedupe/drop-empty; None → NULL) and store it (D1, D3)
- [x] 2.2 Write each `CaseResult.error` into the `case_results.error` column during `record_run` (loop behavior unchanged — the runner already captures errors in memory)
- [x] 2.3 Update `cli.evaluate` to accept `--tags <a,b>` and pass the parsed list to `record_run`
- [x] 2.4 Update the optimize path (`build_objective`/`run_study` callers) to accept and thread `--tags` into every trial run's `record_run`

## 3. Tag filter in read-back

- [x] 3.1 Add optional `tag=None` param to `LineageDB.list_runs` filtering after row assembly (`tag in row.tags.split(",")`), returning empty list on no match (D4)
- [x] 3.2 Add `--tag <tag>` flag to `cli.show`, `cli.compare`, and `cli.export`; pass through to `list_runs`
- [x] 3.3 No matching runs → print "no runs found matching the tag" (and for compare, "comparison needs at least two matching runs")

## 4. Compare: config delta section

- [x] 4.1 Implement a config-delta helper (added/removed/changed keys vs baseline config) on the compare data path
- [x] 4.2 Add `--diff-config` flag to `cli.compare`; when set, render the delta section next to metric deltas (equal configs → no-change marker) (D5)

## 5. Compare: per-case regression table

- [x] 5.1 Add `--cases` flag to `cli.compare`; pair baseline (explicit `--baseline` or predecessor of newest) vs newest run by case index (D6)
- [x] 5.2 Implement the paired per-case table in `src/crucible/reporting/render.py` (case id, before, after, Δ) plus "N broke / M fixed / K unchanged" summary (broke = score decreased, fixed = increased, unchanged = equal)
- [x] 5.3 Print a dataset_version mismatch warning when the two runs' versions differ; pair up to the shorter case list when lengths differ
- [x] 5.4 Handle the degenerate cases: fewer than two runs → "needs two runs" message; a run with no case results → "per-case comparison not possible"

## 6. Error visibility in show

- [x] 6.1 Render an `error` column in `cases_table` (blank for clean cases) in `src/crucible/reporting/render.py`
- [x] 6.2 `show --run` prints "N cases errored" summary when N > 0

## 7. Tests

- [x] 7.1 Hermetic DB tests: backfill adds `tags`/`error` to a pre-existing database without losing rows; new databases get the columns
- [x] 7.2 Hermetic DB tests: `record_run` stores normalized tags and per-case errors; `list_runs(tag=...)` filters correctly
- [x] 7.3 CLI tests: `evaluate --tags a,b` records tags; `show/compare/export --tag` filter and no-match messages (StubLLMClient)
- [x] 7.4 CLI tests: `compare --diff-config` delta section and no-change marker
- [x] 7.5 CLI tests: `compare --cases` table + summary line + dataset_version warning
- [x] 7.6 CLI tests: `show --run` renders error column and errored count
- [x] 7.7 Full suite green: `uv run pytest tests/ -q` (119 existing + new)
