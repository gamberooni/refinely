# Design: config-loop

## Context

The evaluate → optimize → adopt loop currently ends in copy-paste: `optimize` prints a best config as JSON that the user must re-type into `evaluate --config`, and the model under test is a single global (`settings.model_name`) used by both app calls and LLM judging. The codebase already has the building blocks: per-app registrations with `default_config`, a `--config` inline-JSON merge in `cli.evaluate`, configs already serialized into `evaluation_runs.configuration`, and a backfill-aware `LineageDB.init_schema`.

Key mechanism discovered in code: apps read the model at call time via `self._settings.model_name` (e.g. `QAApp.execute` → `client.chat_structured(self._settings.model_name, ...)`), and the judge is wired at build time (`_metrics_factory` → `LLMJudgeMetric(model=settings.model_name)`). This means the app model can be overridden per-run by passing a *copied* `Settings` with `model_name` changed, without touching configs or the client (the client is built from `api_key`/`base_url` only).

## Goals / Non-Goals

**Goals:**
- Named, versionable per-app configs on disk + a `config` CLI group (save/list/show/rm/default).
- `--config` accepts a name OR inline JSON; no `--config` → per-app default pointer.
- `optimize` auto-saves best config to `configs/<app>/opt-best.json` and prints the path.
- Model becomes an orthogonal per-run axis: `--model` on evaluate/optimize, `--models` fan-out on evaluate, `model_name` column on `evaluation_runs`, model column + `--model` filter on compare.

**Non-Goals:**
- Configs stored in the lineage DB (configs are inputs, belong in the repo).
- Retroactive re-tagging (D2's concern, not here).
- DSPy compile model axis (`dspy_compiles` unchanged).
- Judge model per-run override (stays on `settings` for judging consistency).

## Decisions

### D1: Configs live as JSON files at `configs/<app>/<name>.json`
Named configs are plain JSON files, namespaced per app, in a git-versionable `configs/` dir at repo root (cwd-relative, mirroring the `datasets/` convention). New module `src/refinely/config.py` owns the storage layer: `save_config(app, name, config)`, `list_configs(app=None)`, `show_config(app, name)`, `rm_config(app, name)`, `set_default(app, name)`, `clear_default(app)`, `default_config(app, registered_default)` → resolved effective config.
- **Alternative rejected**: DB table — configs are *inputs* that should diff/commit cleanly with the repo; the DB is for *outputs* (lineage). Editing a config is an editor + git operation, not a SQLite write.

### D2: Default pointer = plain-text file `configs/<app>/.default`
The per-app default is a one-line file containing the config name (missing file / empty → no default). Colocated with the configs it points at, no cross-app coupling, trivially git-versionable.
- **Alternative rejected**: central `configs/.defaults.json` map — single write target creates clobber risk across apps and one more JSON parse.

### D3: Named configs merge over the app's registered defaults — same semantics as inline JSON
`evaluate --config <name>` loads `configs/<app>/<name>.json` and merges it over `registration.default_config`, identical to the existing inline-JSON path. Disambiguation in `--config`: try `json.loads` first — a valid JSON object is inline; anything else is treated as a config name (missing file → clear error). `opt-best.json` is also a partial config (search-space keys only) and merges cleanly under this rule.
- **Alternative rejected**: named configs used verbatim as the full config — diverges from the existing inline-merge contract and would make partial configs (like a single-param save) silently drop defaults.

### D4: Config names must be safe filenames
`config save` validates the name against a strict pattern (letters/digits/`-`/`_`, no path separators, no leading `.`), blocking path traversal and hidden-file collisions (notably `.default`). `rm`/`show` reject the same set. `opt-best` is reserved as an auto-managed name.

### D5: Model axis via a copied `Settings`, never in the config
`cli.evaluate`/`optimize` accept `--model` (default `settings.model_name`). When given, the CLI builds `app_settings = settings.model_copy(update={"model_name": model})` and passes **app_settings** to `build_adapter` (apps read `self._settings.model_name` at call time → app calls use the model) while passing **base settings** to `metrics_factory` (judge keeps `settings.model_name`). `record_run` gains a `model_name` param; the CLI records `app_settings.model_name`. No client rebuild needed (client is model-agnostic).
- **Alternative rejected**: `model` as a config key — the confirmed decision is config files hold prompts/params only; model is an orthogonal axis.
- **Alternative rejected**: re-plumbing a `model` param through every `execute` signature — settings-copy rides existing plumbing (`build_adapter` already receives settings) with zero per-app churn.

### D6: `evaluate --models a,b,c` = sequential fan-out, one recorded run each
The CLI loops the model list; each iteration builds its own app instance (own copied settings), runs the full `EvaluationRunner` pass, records its own run, and prints a per-model panel. A failure in one model propagates (matches single-run semantics — no silent swallowing).

### D7: Schema upgrade reuses the existing backfill pattern
`init_schema`'s `_backfill_columns` gains a second guarded `ALTER TABLE evaluation_runs ADD COLUMN model_name TEXT` (mirroring the existing `metric_scores` backfill). `EvaluationRun` dataclass gains `model_name: str | None = None` so pre-migration rows (NULL) still render. `_row_to_run` picks it up automatically via `EvaluationRun(**data)`.

### D8: `list_runs` gains an optional `model_name` filter; compare renders the column
`LineageDB.list_runs(app_name, limit, offset, model_name=None)` adds a WHERE clause when set. `compare_table` gains a model column; `compare` passes `--model` through. Pre-migration NULLs render blank.

## Risks / Trade-offs

- **[Inconsistent model across app vs judge]** → Deliberate per the confirmed decision (consistent judge while varying app model); documented in the spec deltas so it reads as intended, not a bug.
- **[Name/JSON ambiguity in `--config`]** → Deterministic rule: valid JSON object = inline, else name. A config literally named `{"a":1}` is impossible (braces/colons are not safe filenames) — the rule is unambiguous in practice.
- **[Old DB rows have NULL model_name]** → Dataclass default + blank rendering; backfill never fabricates data it doesn't have.
- **[`opt-best.json` overwritten each optimize run]** → Intended (best-so-far snapshot); `config save` lets users promote it to a durable name (`config save best --app ... --config <path>` is not needed — `--config opt-best` resolves the name; users can copy via `config show`).
- **[Fan-out multiplies cost]** → Sequential by design; user chooses the list explicitly.

## Migration Plan

1. Deploy schema change (backfill handles both new and pre-existing DBs; `metric_scores` backfill precedent exists).
2. New `configs/` directory — created on first `config save` / `opt-best` write; nothing to migrate.
3. No rollback concern beyond removing the column from the insert path; old DBs remain readable (NULL model_name).

## Open Questions

- Config dir location is fixed at cwd-relative `configs/`. If the repo later needs a configurable location (e.g. `settings.configs_dir`), it's a one-flag change — deferred.
- `show`'s run-history table does not render the model column (compare does, per the confirmed scope). Flagging here in case review wants symmetric display.
