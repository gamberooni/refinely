# AGENTS.md

## Project

Refinely: evaluate and optimize LLM application configs (Extraction + retrieval-lite QA + RAG apps). Greenfield prototype, spec-driven via OpenSpec (archived changes in `openspec/changes/archive/`, main specs in `openspec/specs/`).

## Commands

```bash
uv sync --group dev          # install (or: make install)
uv run pytest tests/ -q      # 247 tests, no network (or: make test)
uv run pytest tests/test_metrics.py -q
uv run refinely evaluate <extraction|qa|rag> [--config <name|json>] [--model <name>]
uv run refinely optimize <app> [--trials N] [--model <name>]
uv sync --group dspy         # install optional DSPy dep
uv run refinely compile <extraction|qa|rag> [--max-examples N]
make clean                   # caches + lineage.db
```

## Config / env quirks

- Settings (`src/refinely/core/settings.py`) use env prefix `REFINELY_` + `.env` at repo root, with `OPENAI_API_KEY` fallback and optional `base_url`. Real local setup: gateway `http://localhost:4000/v1`, model `deepseek-v4-flash`. `.env` is gitignored — never commit it.
- The user edits `settings.py` defaults and `.env.example` themselves — don't revert their edits.
- CLI raises `REFINELY_OPENAI_API_KEY is not set` when no key; Settings resolves it, so real-API runs work once `.env` exists.
- **Model is an orthogonal axis, never part of a config file.** `--model` on evaluate/optimize overrides the app model only; the LLM judge always uses base `settings.model_name`. `evaluate --models a,b,c` fans out to one recorded run per model. `model_name` is a column on `evaluation_runs`.
- Named configs live as JSON files at `configs/<app>/<name>.json` (cwd-relative, like `datasets/`), managed via the `config` CLI group. `--config` on evaluate accepts a stored name OR inline JSON (disambiguated by `json.loads`). Per-app default = plain-text pointer `configs/<app>/.default` (`config default <app> --set <name>`). `opt-best` is reserved (optimize auto-saves best trial there). Storage API in `src/refinely/config.py`.

## Test gotchas

- Tests run from repo root: dataset paths are cwd-relative (`"datasets/extraction_v1.json"`) — don't `cd` into tests/.
- Hermetic settings: autouse fixture does `monkeypatch.setitem(Settings.model_config, "env_file", None)`. Use **setitem** — `setattr` fails (model_config is a dict subclass).
- No live API calls: use `StubLLMClient` from `tests/stub_llm.py` (canned pop(0) responses — instantiate with enough entries for every call, incl. LLM judge).
- CLI tests monkeypatch the call-time seam, NOT command modules: `refinely.cli.context._client`, `refinely.cli.context.get_registration`, `refinely.cli.context.run_study` (doctor: `refinely.cli.devtools.run_checks`). Commands resolve helpers through `context.X(...)` at call time so patches take effect.

## Architecture notes

- **CLI is a package** (`src/refinely/cli/`), not a single file: `__init__.py` defines the `main` click group, runs `discover_apps()`, then imports submodules so command decorators see registered apps. Command modules: `evaluate.py`, `optimize.py`, `compile.py`, `config_cmds.py`, `readback.py` (show/compare/export), `devtools.py` (new/doctor/dataset). Shared helpers live in `context.py` (`_client`, `_load_run_context`, `_resolve_run_id`, `_format_counts`). New commands: add a module calling `@main.command()` and import it in `__init__.py` (which also re-exports `_load_run_context`). Entry point `refinely = "refinely.cli:main"`; `python -m refinely.cli` works via `__main__.py`.
- Public app API is sync (`app.execute(input, config) -> Result`); apps are duck-typed — no protocol class. Apps wrap the async `AsyncOpenAIClient` via `asyncio.run(...)`. Exception: `RAGApp` runs its whole pipeline (expansion → retrieval → rerank → generation) inside ONE `asyncio.run` per case, aggregating token usage and latency across calls.
- Structured output goes through the JSON-schema-forced fallback path in `chat_structured` (fence stripping + repair retry).
- `case_results` has an `error` TEXT column (nullable, backfilled) — per-case errors ARE persisted; `show --run` renders it + a "N cases errored" line. `evaluation_runs` carries `model_name` + `tags` (normalized comma-separated). Schema upgrades use guarded `ALTER TABLE ... ADD COLUMN` in `LineageDB._backfill_columns` — never drop/recreate tables.
- Optuna studies are named `refinely_{app}` with `load_if_exists=True` — repeated optimize runs resume the same study.
- Apps self-register via `register_app` (`src/refinely/registry.py`): an `AppRegistration` bundles `build_adapter`, `metrics_factory`, `search_space`, `default_config`, `weights`, `dataset_path`, optional `dspy_factory`. Apps are discovered through entry points in group `refinely.apps`: `discover_apps()` (`registry.py`) loads each entry point module (its import calls `register_app`); `cli.py` calls it at import. New apps: create a module anywhere (in-tree `apps/` — a sibling of `src/`, outside the refinely package — or an external codebase) that calls `register_app` at import, and declare it in `[project.entry-points."refinely.apps"]` (demo apps are declared in refinely's own `pyproject.toml`; a dataset is also needed for demo apps, in `datasets/`). `refinely new app <name>` scaffolds `apps/<name>.py` + a dataset stub (never edits pyproject.toml). Generic retrieval (keyword/hybrid BM25-ish scorer over an in-memory corpus) lives in `apps/common.py`. RAG-specific deterministic metrics (`retrieval_recall`, `citation_accuracy`) live in `apps/rag.py` (exact match in `apps/extraction.py`); `_expected_text` must be used by any metric reading `case.expected` as text (rag expected is a dict with `answer` + `source_indices`).
- Developer tooling (`src/refinely/devtools/`) holds `scaffold.py` (`write_app` templates) and `doctor.py` (deterministic no-network checks + opt-in `--network` probe; non-zero exit on failure). `dataset stats <app>` uses `dataset_stats` in `src/refinely/eval/datasets.py`.
- RAG reranking is skipped when the deterministic retrieval returns ≤1 candidate (no LLM call); `retrieve_snippets` (QA) delegates to `retrieve_snippets_indexed` with the hybrid strategy — keep them consistent.
- DSPy integration is optional (`uv sync --group dspy`). Each app can declare a `dspy_factory` on its `AppRegistration` returning a `DspyProgramSpec` (build / prepare_example / prediction_to_output callables). `refinely compile <app>` runs `BootstrapFewShot` against the app's dataset + registered metrics and saves an artifact JSON. `refinely evaluate --program <path>` loads a compiled artifact at build time. Compile lineage is stored in `dspy_compiles` (separate from `evaluation_runs`); `record_compile` + `best_compile` in `LineageDB`. `litellm<1.92` is pinned in the `dspy` group (1.92+ requires Rust/Cargo for macOS builds).

## Conventions

- Behavior is spec'd in `openspec/specs/<capability>/spec.md` — keep specs in sync when behavior changes (sync workflow via `openspec` CLI).
- No code comments unless asked.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for environment setup, commands, architecture, and testing guidance.
