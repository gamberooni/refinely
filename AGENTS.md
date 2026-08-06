# AGENTS.md

## Project

Crucible: evaluate and optimize LLM application configs (Extraction + retrieval-lite QA + RAG apps). Greenfield prototype, spec-driven via OpenSpec (archived change in `openspec/changes/archive/`, main specs in `openspec/specs/`).

## Commands

```bash
uv sync --group dev          # install (or: make install)
uv run pytest tests/ -q      # 119 tests, no network (or: make test)
uv run pytest tests/test_metrics.py -q
uv run crucible evaluate <extraction|qa|rag>
uv run crucible optimize <app> [--trials N]
uv sync --group dspy         # install optional DSPy dep
uv run crucible compile <extraction|qa|rag> [--max-examples N]
make clean                   # caches + lineage.db
```

## Config / env quirks

- Settings (`src/crucible/core/settings.py`) use env prefix `CRUCIBLE_` + `.env` at repo root, with `OPENAI_API_KEY` fallback and optional `base_url`. Real local setup: gateway `http://localhost:4000/v1`, model `deepseek-v4-flash`. `.env` is gitignored — never commit it.
- The user edits `settings.py` defaults and `.env.example` themselves — don't revert their edits.
- CLI raises `CRUCIBLE_OPENAI_API_KEY is not set` when no key; Settings resolves it, so real-API runs work once `.env` exists.

## Test gotchas

- Tests run from repo root: dataset paths are cwd-relative (`"datasets/extraction_v1.json"`) — don't `cd` into tests/.
- Hermetic settings: autouse fixture does `monkeypatch.setitem(Settings.model_config, "env_file", None)`. Use **setitem** — `setattr` fails (model_config is a dict subclass).
- No live API calls: use `StubLLMClient` from `tests/stub_llm.py` (canned pop(0) responses — instantiate with enough entries for every call, incl. LLM judge).

## Architecture notes

- Public app API is sync (`app.execute(input, config) -> Result`); apps are duck-typed — no protocol class. Apps wrap the async `AsyncOpenAIClient` via `asyncio.run(...)`. Exception: `RAGApp` runs its whole pipeline (expansion → retrieval → rerank → generation) inside ONE `asyncio.run` per case, aggregating token usage and latency across calls.
- Structured output goes through the JSON-schema-forced fallback path in `chat_structured` (fence stripping + repair retry).
- `case_results` table intentionally has no error column — per-case errors exist only in memory.
- Optuna studies are named `crucible_{app}` with `load_if_exists=True` — repeated optimize runs resume the same study.
- Apps self-register via `register_app` (`src/crucible/registry.py`): an `AppRegistration` bundles `build_adapter`, `metrics_factory`, `search_space`, `default_config`, `weights`, `dataset_path`, optional `dspy_factory`. Apps are discovered through entry points in group `crucible.apps`: `discover_apps()` (`registry.py`) loads each entry point module (its import calls `register_app`); `cli.py` calls it at import. New apps: create a module anywhere (in-tree `apps/` — a sibling of `src/`, outside the crucible package — or an external codebase) that calls `register_app` at import, and declare it in `[project.entry-points."crucible.apps"]` (demo apps are declared in crucible's own `pyproject.toml`; a dataset is also needed for demo apps, in `datasets/`). Generic retrieval (keyword/hybrid BM25-ish scorer over an in-memory corpus) lives in `crucible/retrieval.py`. RAG-specific deterministic metrics (`retrieval_recall`, `citation_accuracy`) live in `apps/rag.py` (exact match in `apps/extraction.py`); `_expected_text` must be used by any metric reading `case.expected` as text (rag expected is a dict with `answer` + `source_indices`).
- RAG reranking is skipped when the deterministic retrieval returns ≤1 candidate (no LLM call); `retrieve_snippets` (QA) delegates to `retrieve_snippets_indexed` with the hybrid strategy — keep them consistent.
- DSPy integration is optional (`uv sync --group dspy`). Each app can declare a `dspy_factory` on its `AppRegistration` returning a `DspyProgramSpec` (build / prepare_example / prediction_to_output callables). `crucible compile <app>` runs `BootstrapFewShot` against the app's dataset + registered metrics and saves an artifact JSON. `crucible evaluate --program <path>` loads a compiled artifact at build time. Compile lineage is stored in `dspy_compiles` (separate from `evaluation_runs`); `record_compile` + `best_compile` in `LineageDB`. `litellm<1.92` is pinned in the `dspy` group (1.92+ requires Rust/Cargo for macOS builds).

## Conventions

- Behavior is spec'd in `openspec/specs/<capability>/spec.md` — keep specs in sync when behavior changes (sync workflow via `openspec` CLI).
- No code comments unless asked.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for environment setup, commands, architecture, and testing guidance.
