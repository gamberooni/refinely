# Tasks: dev-ergonomics

## 1. Scaffold

- [x] 1.1 Add `src/crucible/scaffold.py` with app-module and dataset-stub templates following the `apps/extraction.py` convention (DATASET_PATH via `parents[1] / "datasets" / "<name>_v1.json"`, register_app skeleton with TODO placeholders for build_adapter, metrics_factory, search_space, default_config, weights, dataset_path) (D1)
- [x] 1.2 Implement `write_app(name, dataset_path)` with name validation (`str.isidentifier()` + reserved guard) and an existing-target guard that refuses to overwrite (D2)
- [x] 1.3 Add `new app` click subcommand to `cli.py`: `crucible new app <name> [--dataset <path>]`; with `--dataset`, point dataset_path at it and skip the stub; always print the `[project.entry-points."crucible.apps"]` line to add; never edit pyproject.toml (D5)

## 2. Doctor

- [x] 2.1 Add `src/crucible/doctor.py` with a `CheckResult` model and check functions: apps (discover_apps ≥1), datasets (load_dataset per registered app), schema (LineageDB open → init_schema), env (settings.openai_api_key non-empty) (D3)
- [x] 2.2 Add the opt-in network check (`--network`) probing the configured base_url, and a `run_checks(settings, network)` orchestration returning results (D3)
- [x] 2.3 Add `doctor` click subcommand rendering a rich panel per check with fix hints; exit code 0 all-pass / 1 any-fail; no network calls without `--network` (D3, D5)

## 3. Dataset stats

- [x] 3.1 Add `dataset_stats(path) -> DatasetStats` to `src/crucible/eval/datasets.py`: case count, file_size_bytes, input-key presence counts, expected type/key histogram, malformed-case report (modal-shape deviation), never raising on structural inconsistencies; parse errors propagate (D4)
- [x] 3.2 Add `dataset stats` click subcommand: resolve the app's dataset_path via `_load_run_context`, render the stats panel, surface parse errors naming file + failing case (D4, D5)

## 4. Tests

- [x] 4.1 Scaffold tests: writes both files, `--dataset` variant, invalid name errors, existing-file refusal (use tmp_path, never the real apps/ dir)
- [x] 4.2 Doctor tests: all-pass → exit 0; per-failing-check fix hints → exit 1 (monkeypatch each check); no network without `--network`
- [x] 4.3 Dataset stats tests: counts/shapes/malformed report on a crafted dataset; missing/invalid file error propagation
- [x] 4.4 CLI tests for `new app`/`doctor`/`dataset stats` via CliRunner with stubbed context
- [x] 4.5 Full suite green: `uv run pytest tests/ -q` (119 existing + new)
