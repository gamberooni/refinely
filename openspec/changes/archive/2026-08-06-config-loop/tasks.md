# Tasks: config-loop

## 1. Config storage layer

- [x] 1.1 Add `src/refinely/config.py` with the `configs/<app>/<name>.json` storage functions (save/list/show/rm) plus path helpers, following the cwd-relative `datasets/` convention
- [x] 1.2 Implement name validation (strict `[A-Za-z0-9_-]+`, no path separators, no leading `.`, `opt-best` reserved) enforced on save/show/rm
- [x] 1.3 Implement the default pointer (`configs/<app>/.default` plain-text file): `set_default`, `clear_default`, `get_default`, with `rm` clearing the pointer when it pointed at the removed config
- [x] 1.4 Implement `default_config(app, registered_default)`: no pointer → registered default; pointer set → named config merged over registered default

## 2. Config CLI group

- [x] 2.1 Add the `config` subcommand group to `src/refinely/cli.py`: `save <name> --app <app> --config <json>`, `list [--app]`, `show <name> --app`, `rm <name> --app`, `default <app> --set <name>` / `--clear`
- [x] 2.2 `config save` validates JSON object + name, writes the file, reports the path; invalid input exits with a clear error and no file created
- [x] 2.3 `config list` marks the default config with a star; `config show` prints the file contents; `config rm` deletes the file (and clears the default pointer if needed)

## 3. Config resolution in evaluate/optimize

- [x] 3.1 In `cli.evaluate`, resolve `--config`: valid JSON object → inline merge (existing behavior); otherwise treat as name → load `configs/<app>/<name>.json` and merge over `registration.default_config`; unknown name → clear error
- [x] 3.2 In `cli.evaluate`, no `--config` → use `default_config(app, registration.default_config)` (pointer-aware)
- [x] ~~3.3 Apply the same name-or-inline resolution and default fallback in `cli.optimize`~~ — **out of scope** (user decision): optimize samples configs from the search space; `--config` resolution is evaluate-only

## 4. Model axis

- [x] 4.1 Add `--model <name>` flag to `evaluate` and `optimize` (default `settings.model_name`)
- [x] 4.2 When `--model` is given, build the app with `settings.model_copy(update={"model_name": model})` while metrics/judge keep base settings (D5)
- [x] 4.3 Add `model_name TEXT` column to `evaluation_runs` metadata + guarded `ALTER TABLE` backfill in `LineageDB._backfill_columns` (D7)
- [x] 4.4 Add `model_name: str | None = None` to `EvaluationRun` dataclass; `_row_to_run` picks it up automatically
- [x] 4.5 Add `model_name` param to `LineageDB.record_run` and thread it through from `cli.evaluate` and the optimizer objective (`build_objective`/`run_study` path)
- [x] 4.6 Add `evaluate --models a,b,c` fan-out: one sequential run per model, each recorded with its own model_name, one result panel per model; empty list → clear error

## 5. Optimize auto-save

- [x] 5.1 After `run_study` completes with a best trial, write the best trial's config to `configs/<app>/opt-best.json` (overwrite) and include the path in the optimize output panel
- [x] 5.2 No successful trials → clear error and no `opt-best.json` written

## 6. Read-back: model in compare

- [x] 6.1 Add optional `model_name` filter param to `LineageDB.list_runs` (WHERE clause when set)
- [x] 6.2 Add model column to `compare_table` in `src/refinely/reporting/render.py`; NULL renders blank
- [x] 6.3 Add `--model <name>` flag to `cli.compare`, pass through to `list_runs`, no-matches → "no runs found for that model"

## 7. Tests

- [x] 7.1 Tests for config storage layer (save/list/show/rm, name validation, default pointer, merge semantics) using `tmp_path`
- [x] 7.2 CLI tests for the `config` group (save/list/show/rm/default) via the click runner
- [x] 7.3 CLI tests for `--config` name resolution + unknown-name error + default pointer fallback
- [x] 7.4 Tests for `--model` recording (recorded model_name), `--models` fan-out (one run per model), and `model_name` backfill on a pre-migration schema
- [x] 7.5 Tests for optimize auto-save (`opt-best.json` written, path printed, no-success → no file)
- [x] 7.6 Tests for compare model column + `--model` filter (including blank NULL rendering)
- [x] 7.7 Run full suite (`uv run pytest tests/ -q`) — all 119 existing + new tests green
