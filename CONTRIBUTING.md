# Contributing to Refinely

Welcome! This document is a reference for anyone working on Refinely — what it does, how to set it up, how the pieces fit together, and the traps to avoid. It was written from the actual codebase, so if you find something here that disagrees with the code, the code wins — and please fix this document.

## Project Overview

Refinely is a minimal prototype for evaluating and optimizing LLM application configurations. It is a Python 3.11 CLI package (built with uv/hatchling, distributed as the `refinely` console script) that:

1. Runs a configurable "application" (an extraction app, a retrieval-lite QA app, or a 4-stage RAG app) over a versioned JSON dataset of cases.
2. Scores every case with a set of weighted metrics (exact/fuzzy match, LLM judge, retrieval recall, citation accuracy, latency, cost).
3. Records every run — including per-case inputs, outputs, expected values, and scores — into a SQLite lineage database.
4. Uses Optuna (TPE sampler) to search a per-app configuration space, recording each trial into the same SQLite database so runs and trials are directly comparable.

The project was built as a greenfield, spec-driven prototype via OpenSpec (see `openspec/` — completed changes are archived under `openspec/changes/archive/`, and the resulting capability specs live in `openspec/specs/`). It is intentionally small: three toy apps, real LLM calls, and a deliberately simple architecture that makes the evaluation/optimization loop easy to trace end to end.

The intended consumer is the CLI. There is no HTTP server, no web UI, and no library API contract — apps plug in as plain objects with `execute(input, config) -> Result` (duck-typed, no protocol class) plus a `register_app` registration, so new applications can be added without touching the evaluation machinery.

## Environment Setup

### Prerequisites

- **Python 3.11+** (declared via `requires-python = ">=3.11"` in `pyproject.toml`)
- **uv** — the project uses uv for environment and dependency management. If you don't have it, install per [the uv docs](https://docs.astral.sh/uv/). There is no `requirements.txt` or `setup.py`; uv reads `pyproject.toml` directly.

### Install

```bash
uv sync --group dev
```

This creates `.venv/`, installs runtime dependencies (`openai`, `pydantic`, `pydantic-settings`, `tenacity`, `optuna`, `click`, `sqlalchemy>=2.0`, `rich`) plus the dev group (`pytest`), and installs the package itself in editable-ish form so the `refinely` console script is available via `uv run`.

### Configure the environment

Copy the template and fill in your values:

```bash
cp .env.example .env
```

The variables are documented in [Configuration System](#configuration-system) below. The two things you almost always need are `REFINELY_OPENAI_API_KEY` (or the `OPENAI_API_KEY` fallback) and, if you use a local gateway, `REFINELY_BASE_URL`.

> **Security note:** `.env` is gitignored — never commit it. `.env.example` contains placeholders only.

### Verify setup

```bash
uv run pytest tests/ -q    # expect: 247 passed, no network calls
uv run refinely --help     # CLI renders
```

## Build & Run Commands

### Test

```bash
uv run pytest tests/ -q              # full suite (247 tests, ~1s, no live API)
uv run pytest tests/test_metrics.py -q   # one file
```

Linting is ruff (`[tool.ruff] line-length = 100`, `extend-select = ["I"]` for import sorting in `pyproject.toml`). There is no CI gate, so keep both the test suite and `uv run ruff check src apps tests` green before committing.

### Run the CLI

```bash
uv run refinely evaluate extraction
uv run refinely evaluate qa
uv run refinely evaluate rag
uv run refinely optimize extraction --trials 15
uv run refinely optimize qa --trials 3
uv run refinely optimize rag --trials 3
# optional DSPy compile (requires uv sync --group dspy)
uv run refinely compile extraction --max-examples 20
```

- `evaluate <app>` runs the app's registered default config (`register_app` in `apps/*.py`) over its dataset and records one lineage run. `--config <name|json>` uses a stored config (`configs/<app>/<name>.json`) or an inline JSON object (merged over the app's defaults; no `--config` uses the per-app default pointer). `--model <name>` overrides the app model only (the judge stays on `settings.model_name`); `--models a,b,c` fans out to one recorded run per model. `--tags a,b` tags the run.
- `optimize <app> [--trials N]` runs an Optuna study (default 15 trials), records one lineage run per trial, and auto-saves the best trial's config to `configs/<app>/opt-best.json`.
- Named configs are managed via the `config` group: `config save <name> --app <app> --config '{"..."}'`, `config list`, `config show <name> --app`, `config rm <name> --app`, `config default <app> --set <name>|--clear`.
- Read-back: `show`/`compare`/`export` accept `--tag` to filter; `compare` adds `--diff-config` (config delta vs. baseline) and `--cases` (per-case broke/fixed/unchanged drilldown); `show --run` renders persisted per-case errors + an "N cases errored" line.
- Developer tooling: `new app <name>` scaffolds `apps/<name>.py` + `datasets/<name>_v1.json` (prints the pyproject entry point to add, never edits `pyproject.toml`); `doctor` runs deterministic health checks (`--network` adds a gateway probe); `dataset stats <app>` reports case counts, shapes, and malformed cases.

### Makefile

```bash
make install                 # uv sync --group dev
make test                    # uv run pytest tests/ -q
make evaluate APP=extraction
make optimize APP=qa TRIALS=15
make clean                   # removes .pytest_cache, .coverage, __pycache__, and lineage.db
```

## Project Layout

```
pyproject.toml               # deps, [project.scripts] refinely, [project.entry-points."refinely.apps"], hatchling, ruff, pytest config
apps/                        # demo apps — sibling of src/, outside the refinely package
  __init__.py                # imports demo app modules (registration side effects)
  common.py                  # generic keyword/hybrid retrieval (retrieve_snippets, retrieve_snippets_indexed), shared by QA + RAG
  extraction.py              # ExtractionApp + ExactMatchMetric + search space/defaults/weights (register_app)
  qa.py                      # QAApp + search space/defaults/weights (register_app)
  rag.py                     # RAGApp pipeline + retrieval/citation metrics + search space/defaults/weights (register_app)
configs/                     # named configs as JSON files: configs/<app>/<name>.json + opt-best.json + .default pointer
src/refinely/
  cli/                       # CLI package (click group): main + subcommand modules
    __init__.py              # main group, discover_apps, submodule imports, re-exports _load_run_context
    context.py               # shared helpers (_client, _load_run_context, _resolve_run_id, _format_counts) — call-time monkeypatch seam
    evaluate.py              # evaluate (+ _resolve_config, _run_evaluation)
    optimize.py              # optimize
    compile.py               # compile
    config_cmds.py           # config group: save / list / show / rm / default
    readback.py              # show / compare / export (+ _compare_pair, _compare_diff_config, _compare_cases)
    devtools.py              # new app / doctor / dataset
    __main__.py              # python -m refinely.cli support
  config.py                  # named config storage: configs/<app>/<name>.json, .default pointer, opt-best (ConfigError)
  data.py                    # bundled_dataset — resolves demo datasets from the wheel (refinely/datasets) or repo-root datasets/
  registry.py                # AppRegistration + register_app / get_registration / registered_apps / discover_apps
  devtools/
    scaffold.py              # write_app — templates for apps/<name>.py + dataset stub (ScaffoldError)
    doctor.py                # CheckResult + run_checks (apps/datasets/schema/env + opt-in network) — no network by default
  core/
    exceptions.py            # RefinelyError base; LLMError, EvalError (module-specific subclasses like ConfigError/ScaffoldError live with their domain code)
    settings.py              # pydantic-settings, REFINELY_ env prefix, .env loading
  llm/
    client.py                # AsyncOpenAIClient: chat_text / chat_structured
    usage.py                 # TokenUsage model (prompt/completion tokens)
  eval/
    datasets.py              # EvalCase model, load_dataset, load_corpus, dataset_version, dataset_stats (DatasetStats)
    metrics.py               # generic metrics (fuzzy, llm_judge, latency, cost), Metric, aggregate_scores
    runner.py                # EvaluationRunner, CaseResult, EvaluationRunResult
  optimize/
    objective.py             # build_objective — wraps eval + lineage in a trial (registry defaults)
    study.py                 # run_study — create/resume + optimize
  tracking/
    db.py                    # LineageDB — SQLAlchemy Core schema + record/query helpers (evaluation_runs + dspy_compiles)
    models.py                # EvaluationRun / CompileRecord / CaseRecord pydantic read models
  reporting/
    render.py                # runs_table / cases_table / compare_table / case_pair_table / config_delta / best-run + best-compile panels
    export.py                # export_runs_csv / export_runs_json
  dspy/
    spec.py                  # DspyProgramSpec dataclass (build / prepare_example / prediction_to_output)
    bridge.py                # metric bridge (example_case, make_dspy_metric, score_result)
    compile.py               # compile_program + CompileResult + _split_train_val
    lm.py                    # configure_lm — wire dspy.LM from Settings
    _imports.py              # lazy _dspy() helper (ImportError → clear EvalError)
datasets/
  extraction_v1.json         # versioned dataset: version + cases
  qa_v1.json                 # versioned dataset: version + corpus + cases
  rag_v1.json                # versioned dataset: version + corpus + cases (expected has source_indices)
tests/
  stub_llm.py                # StubLLMClient (canned responses, no network)
  test_*.py                  # one file per module
openspec/                    # OpenSpec changes (archive/) and main capability specs (specs/)
```

Organizing principle: `src/refinely/` mirrors the evaluation pipeline — the app layer (what gets evaluated), the eval layer (how it's scored), the optimize layer (how configs are searched), and the tracking layer (where everything lands). Apps are duck-typed objects (`execute(input, config) -> Result`, no protocol class); metrics are decoupled from the runner via the `Metric` protocol. Each app bundles its metric set, weights, search space, and default config into an `AppRegistration` via `register_app` (`registry.py`), and apps are discovered through entry points in group `refinely.apps` (`discover_apps`), so the framework core stays app-agnostic and apps can live outside the refinely repo. Named configs are versionable JSON files under `configs/` (cwd-relative, like `datasets/`), managed by the `config` CLI group through the storage API in `config.py`.

## Architecture

The core loop is: **sample a config → execute the app over the dataset → score with weighted metrics → record to lineage → (optimization) feed the aggregate score back to Optuna**.

```
                config dict                    config dict
CLI ──┬─────► evaluate: ──► EvaluationRunner ──► app.execute(input, config)
      │                          │                        │
      │                          │                        └─► AsyncOpenAIClient (async) via asyncio.run
      │                          │                        └─► returns Result(output, token_usage, latency)
      │                          ▼
      │                    Metrics (per case) ─► aggregate_scores (weighted)
      │                          │
      │                          ▼
      │                    LineageDB.record_run ─► evaluation_runs / metric_results / case_results
      │
      └─────► optimize: ──► build_objective ──► (loop above, once per trial)
                              │
                              └─► run_study: Optuna TPE, storage = sqlite:///lineage.db
                                     study name "refinely_{app}", load_if_exists=True (resumes)
```

### Key abstractions

- **Apps** (duck-typed, no protocol class) — any object exposing `execute(input: dict, config: dict) -> Result`. This is the seam where new applications plug in. `Result` (`llm/usage.py`) carries `output` (dict or str), `token_usage`, and `latency_seconds`.
- **`Metric`** (`eval/metrics.py`) — Protocol: `evaluate(case, output) -> MetricResult`. The framework ships four generic implementations (fuzzy match, LLM judge, latency, cost); app-specific metrics (exact match, retrieval recall, citation accuracy) live in the app modules that register them. A failing metric scores 0.0 and never aborts the run. Retrieval metrics read `expected["source_indices"]` / `output["retrieved_indices"]` / `output["cited_indices"]`; any metric that treats `case.expected` as text (fuzzy, judge) must use the `_expected_text` helper, because RAG expectations are dicts.
- **`EvaluationRunner`** (`eval/runner.py`) — the per-case execution loop. Any `app.execute` failure is caught per case, the case is recorded with `error` set and zero scores, and the run continues. Errors are persisted on `case_results.error` (nullable; clean cases are `NULL`).
- **`LineageDB`** (`tracking/db.py`) — SQLAlchemy Core over a SQLite file. Tables: `evaluation_runs` (run_id, app, dataset version, JSON config, model_name, tags, optuna trial number, aggregate score), `metric_results`, `case_results` (incl. nullable `error` column), and `dspy_compiles` (compile_id, optimizer, baseline/compiled scores, artifact path). Schema creation is idempotent, deliberately coexists with Optuna's own tables in the same SQLite file, and upgrades are additive only: `_backfill_columns` runs guarded `ALTER TABLE ... ADD COLUMN` for columns added since the DB was created (`model_name`, `tags`, `error`, `metric_scores`) — never drop/recreate.
- **`build_objective`** (`optimize/objective.py`) — the bridge between Optuna and evaluation: samples a config from the trial, runs the evaluation, records lineage with the trial number, returns the aggregate score.
- **`DspyProgramSpec`** (`dspy/spec.py`) — declares a DSPy program per app: three callables (`build`, `prepare_example`, `prediction_to_output`). Apps that set `dspy_factory` on their `AppRegistration` return one of these. `compile_program` (`dspy/compile.py`) uses it to run `BootstrapFewShot` against the app's registered metrics (via the metric bridge in `dspy/bridge.py`) and saves the compiled artifact JSON. The DSPy group is optional (`uv sync --group dspy`); `_dspy()` (`dspy/_imports.py`) is a lazy helper that raises a clear `EvalError` if dspy is absent.

### Flow trace (primary path: `refinely evaluate extraction`)

1. `cli.evaluate` loads the dataset (`datasets/extraction_v1.json`), pulls `default_config`, `weights`, and `metrics_factory` from the `extraction` registration (`registry.get_registration`), and builds `EvaluationRunner(registration.metrics_factory(client, settings), app)`.
2. `EvaluationRunner.run` iterates cases; for each, `ExtractionApp.execute` builds messages from the config (temperature, prompt variant), calls `client.chat_structured` via `asyncio.run(...)`, and wraps it with a `perf_counter` timer.
3. Each metric scores the case; per-metric means become `metric_results`; `aggregate_scores` combines them with the app's weight scheme.
4. `LineageDB.record_run` writes one row per table and commits.

The sync-over-async boundary is worth noting: everything public is synchronous, and async client calls are driven with `asyncio.run(...)` at the app layer. This keeps the CLI, runner, and tests single-threaded and simple. `RAGApp` is the exception in shape, not in kind: its whole pipeline (expansion → retrieval → rerank → generation, up to 4 LLM calls) runs inside a single `asyncio.run` per case so token usage and latency can be aggregated across the stage calls.

## Configuration System

All runtime configuration flows through `Settings` (`core/settings.py`), a pydantic-settings `BaseSettings` with `env_prefix = "REFINELY_"` and `env_file = ".env"` (loaded from the current working directory — normally the repo root).

| Variable | Type | Default | Required | Description |
|---|---|---|---|---|
| `REFINELY_OPENAI_API_KEY` | str | falls back to `OPENAI_API_KEY` env var, else `""` | for real runs | API key for the LLM provider |
| `REFINELY_BASE_URL` | str | `None` | no | OpenAI-compatible base URL; use for a local gateway (e.g. `http://localhost:4000/v1`) |
| `REFINELY_MODEL_NAME` | str | `deepseek-v4-flash` | no | Default model for app calls and LLM judging; per-run override via `evaluate|optimize --model` (judge is unaffected) |
| `REFINELY_LINEAGE_DB_PATH` | str (path) | `lineage.db` | no | SQLite file for lineage + Optuna storage (cwd-relative) |

Precedence: explicit `REFINELY_*` env vars beat `.env` file values, which beat class defaults; `OPENAI_API_KEY` is only consulted when `REFINELY_OPENAI_API_KEY` is unset. The CLI refuses to construct a real client without a key (`REFINELY_OPENAI_API_KEY is not set`), but tests never need one — they use `StubLLMClient`.

The default `model_name` (`deepseek-v4-flash`) and the `.env.example` contents are tuned to a local gateway setup; the user edits these directly, so don't "fix" them back to generic values.

## Local Development

The inner loop is: **edit → `uv run pytest tests/<file> -q` → repeat**. There is no dev server, no hot reload, and no build step — just a Python package installed into `.venv`.

To exercise the CLI against the real backend, you need a reachable provider:

- **OpenAI API**: set `REFINELY_OPENAI_API_KEY` (or export `OPENAI_API_KEY`).
- **Local gateway**: start it, then set `REFINELY_BASE_URL=http://localhost:4000/v1` in `.env`. This is the setup the repo is currently tuned for.

There is no mocking layer in the runtime path; the deterministic, network-free experience is provided by `tests/stub_llm.py` (see Testing Strategy), and that is the expected way to develop most features.

## Using refinely in your codebase

Refinely is consumed as a framework from other repositories (editable path dependency) or extended in-tree via the app registry. See [docs/integration.md](docs/integration.md) for the two usage patterns:

- **Registry app** — `register_app` in a module under `apps/` (sibling of `src/`, outside the refinely package); registration bundles the adapter, metrics factory, search space, default config, weights, and dataset path. The CLI choices and `build_objective` defaults resolve from the registry.
- **Library driver** — your own script passes `app`, `metrics`, `search_space`, and `weights` to `build_objective` explicitly; refinely never touches your settings or runtime.

Keep `docs/integration.md` in sync when the public API surface changes.

## Code Style & Conventions

- **Python 3.11, typed**: all public functions carry annotations, pydantic v2 models (`BaseModel`) are used for structured data, and protocols are used for seams.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for module constants (price constants, budgets, `EXTRACTION_WEIGHTS`/`QA_WEIGHTS`/`RAG_WEIGHTS`, `SYSTEM_PROMPTS`).
- **No code comments unless asked** — the project convention is that code should be self-explanatory; comments are added deliberately, not routinely.
- **Exceptions**: a small hierarchy in `core/exceptions.py` (`RefinelyError` base; `LLMError` for client failures, `EvalError` for evaluation/data/optimize failures). Raise these rather than bare `Exception`s; module-specific subclasses (`ConfigError` in `config.py`, `ScaffoldError` in `devtools/scaffold.py`) live with their domain code, not in `core/exceptions.py`.
- **Imports**: stdlib first, then third-party, then local (`apps.*`, `refinely.*`) — plain alphabetical within groups. ruff's `I` rule set (in `pyproject.toml`) enforces this; run `uv run ruff check src apps tests` to catch sorting drift.
- **User-edited files**: `core/settings.py` defaults and `.env.example` are maintained by hand; treat them as user-owned and don't revert their values.

## Design Patterns & Techniques

- **Duck-typed apps + protocol-based metric seams** (`eval/metrics.py`): apps are plain objects with `execute(input, config) -> Result` (no protocol class); `Metric` is a runtime-checkable `Protocol`. Anything satisfying the signatures works — including test doubles. This is how new apps and metrics plug in without touching the runner.
- **Sync facade over async client**: `AsyncOpenAIClient` is async; apps and metrics call it inside `asyncio.run(...)`. If you add a new app or metric, keep this pattern — do not make the public API async.
- **JSON-schema-forced structured output**: `chat_structured` never relies on `response_format`; it sends the schema in the prompt, then extracts JSON via a fallback path (fence stripping, prose extraction, one repair retry) in `_chat_structured_fallback`. Tenacity retries are wired onto the client methods in `__init__`.
- **Canned-response stubs** (`tests/stub_llm.py`): `StubLLMClient` pops canned responses from queues (`structured_responses`, `text_responses`). Instantiate it with enough entries for every call the code under test will make — including LLM judge calls in QA metric tests.
- **Versioned dataset wrapper**: datasets are `{"version": "...", "cases": [...]}` (QA also has `"corpus"`); `load_dataset` accepts a bare JSON list too, and `dataset_version` falls back to the filename stem. When adding a dataset, point the app's registration `dataset_path` at it (`apps/*.py`).

## Testing Strategy

- **Runner**: pytest, 247 tests, no network access — every test uses `StubLLMClient` with canned responses. Run everything with `uv run pytest tests/ -q` from the repo root.
- **Location/naming**: `tests/test_<module>.py`, mirroring `src/refinely/`. Shared fixture code lives in `tests/stub_llm.py`; each test file defines its own local doubles (e.g. `_StubApp` in `test_runner.py`).
- **Isolation**: tests are hermetic with respect to the repo's `.env`. The autouse fixture in `test_core_settings.py` does `monkeypatch.setitem(Settings.model_config, "env_file", None)` — note `setitem`, because `Settings.model_config` is a dict subclass and `setattr` fails. Other test files rely on the stub client and never construct real `Settings`-driven clients.
- **What's covered**: settings loading (env, fallback, .env precedence), client text/structured paths and JSON extraction helpers, all three apps + shared retrieval in `apps/common.py` (including the strategy switch and indexed variant), dataset loading + error paths + `dataset_stats`, the generic metrics + weight aggregation, the app registry (round-trip, duplicate/unknown errors, search-space/default/weight consistency), the runner (including failure tolerance), lineage round-trips (incl. `model_name`/`tags`/`error` columns + guarded backfills), named-config storage + `config` CLI group, the CLI package commands (evaluate/optimize/show/compare/export incl. `--tag`/`--diff-config`/`--cases`), the developer tools (`new app` scaffold, `doctor`, `dataset stats`), Optuna objective/study (against a tmp_path SQLite file), and the DSPy harness (bridge, split, compile_program end-to-end with stubbed dspy module, lm wiring).
- **CLI test seam**: commands resolve helpers through `refinely.cli.context.X(...)` at call time, so tests monkeypatch `refinely.cli.context._client`, `refinely.cli.context.get_registration`, `refinely.cli.context.run_study`, etc. — never the command module's own binding (e.g. `refinely.cli.devtools.run_checks` for doctor).
- **Expensive tests**: the Optuna study test runs 3 trials × 10 cases against a stub — it takes a few seconds, which is fine, but be aware the suite isn't all sub-second.

## Gotchas & Sharp Edges

- **Tests must run from the repo root.** Dataset paths are cwd-relative (`"datasets/extraction_v1.json"`). Running pytest from `tests/` breaks the suite.
- **`monkeypatch.setitem`, not `setattr`**, for `Settings.model_config` — `model_config` is a dict subclass; `setattr` raises.
- **`StubLLMClient` is queue-based**: a queue that runs dry raises. Count every LLM call (including judge calls) before constructing the stub.
- **Optuna studies resume**: `run_study` uses `load_if_exists=True` with study names `refinely_{app}`. Re-running `optimize` continues the same study — trial numbers keep counting up and `best_trial` reflects the whole history. This is by design (lineage), not a bug.
- **Per-case errors are persisted**: `case_results.error` (nullable TEXT) holds per-case errors — clean cases are `NULL`. `show --run` renders the column and an "N cases errored" line. Schema upgrades are additive-only: `_backfill_columns` issues guarded `ALTER TABLE ... ADD COLUMN` for `model_name`/`tags`/`error`/`metric_scores` on pre-existing DBs; never drop or recreate tables (that would also clobber Optuna's tables in the same file).
- **`lineage.db` is gitignored** (along with `.env`, `.venv`, `.opencode`). It accumulates real data at the repo root; `make clean` removes it. Don't commit it.
- **The lineage DB may contain dead runs**: the manual verification era produced a few baseline runs scoring 0.0 (from a bad API key). Query with `ORDER BY created_at` and filter if you're comparing baselines.
- **Scores are clamped to 0.0–1.0** by construction (weighted means of metric scores); the QA/RAG `fuzzy_match` and `llm_judge` scores run very close to 1.0 on the current datasets, so small deltas are noise — judge config quality by retrieval/citation/latency too. On the RAG dataset, `retrieval_recall` and `citation_accuracy` sit at 1.0 for the baseline (the deterministic matcher finds both sources and the gateway cites them), so their weight only bites when a config hurts retrieval.
- **RAG stage toggles change LLM call counts**: `query_expansion` and `rerank` each add an LLM call per case (rerank only when retrieval returns >1 candidate). Stub queues in RAG tests must account for every stage call plus the judge call.
- **Gateway dependency**: real runs need the local gateway (or a valid OpenAI key). If the gateway is down, every case records an error and the run scores 0.0 — the pipeline doesn't fail loudly; check `case_results`/`metric_results` before trusting a score.
