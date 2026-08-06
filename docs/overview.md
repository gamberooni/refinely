# Crucible — Overview

## What is Crucible

Crucible is a framework for **evaluating and optimizing LLM application configurations**. It runs a configurable LLM app over a versioned evaluation dataset, scores the results with a weighted set of metrics, records every run in a SQLite lineage database, and uses Optuna to search for better configurations automatically.

It is a greenfield prototype that currently ships three toy applications, deliberately chosen to stress the framework in increasing order of complexity:

| App | Pipeline | LLM calls per case | Config params |
|---|---|---|---|
| Extraction | single structured call | 1 | 2 |
| QA | deterministic retrieval + structured call | 1 | 3 |
| RAG | optional query expansion → deterministic retrieval → optional reranking → generation | 1–4 (config-dependent) | 6 |

The RAG app exists specifically to stress-test the framework: it has conditional stages, multiple LLM calls per case, dataset-aware metrics (retrieval recall, citation accuracy), and a mixed-type search space.

## Purpose

- **Diagnose and compare configurations**: answer "which prompt/temperature/top_k actually moves the needle?" with reproducible, queryable experiment history instead of ad-hoc manual testing.
- **Optimize without manual grid search**: let Optuna's TPE sampler explore each app's search space and record the lineage of every trial.
- **Validate the framework itself**: the app adapter protocol, metric plugin system, weight schemes, search space registry, and lineage schema are exercised by increasingly complex apps so that design gaps surface early.

## Goals

- Every evaluation run and every optimization trial is persisted with its configuration, per-metric results, and per-case scores.
- Score computation is **failure-tolerant**: a failing case or metric scores 0.0 and never aborts the run.
- Adding a new app requires only registration entries (metrics, search space, CLI, dataset), not changes to the core engine.
- The full test suite (247 tests) runs without any network access, using a canned-response stub LLM client.
- Behavior is spec-driven: each capability is documented in `openspec/specs/` and changes go through the OpenSpec change workflow.

## In scope

- Three apps behind the duck-typed `execute(input, config) -> Result` contract: `ExtractionApp`, `QAApp`, `RAGApp`.
- Deterministic retrieval (keyword / hybrid scoring, in-memory corpus, no vector store) shared by QA and RAG.
- Seven metrics: `exact_match`, `fuzzy_match`, `llm_judge`, `latency`, `cost`, `retrieval_recall`, `citation_accuracy`; per-app weight schemes that sum to 1.0.
- Weighted aggregate scoring across cases (`aggregate_scores`).
- Optuna-based optimization: per-app search spaces, TPE sampler, SQLite storage shared with the lineage database, resumable studies (`crucible_{app}`, `load_if_exists=True`).
- SQLite lineage: `evaluation_runs` (config, aggregate score, per-run `model_name` and `tags`, optional trial number), `metric_results`, `case_results` (with a nullable persisted `error` column for per-case failures); `dspy_compiles` for compile lineage (separate from evaluation runs). Existing databases are upgraded in place with guarded `ALTER TABLE ... ADD COLUMN` backfills — never dropped or recreated.
- CLI, all rendered with `rich` tables/panels, plus Makefile targets:
  - `crucible evaluate <app> [--config <name|json>] [--model <name>] [--models a,b,c] [--tags a,b] [--program <path>]` — run a single evaluation, or fan out across models (`--models`).
  - `crucible optimize <app> [--trials N] [--model <name>] [--tags a,b]` — Optuna search; the best trial's config is auto-saved to `configs/<app>/opt-best.json`.
  - `crucible compile <app> [--max-examples N]` — optional DSPy behavior optimization.
  - `crucible show <app> [--run <run_id>] [--page N] [--pager] [--tag ...]` — run history and per-case results (with error column + "N cases errored").
  - `crucible compare <app> [--baseline <run_id>] [--model ...] [--tag ...] [--diff-config] [--cases] [--pager]` — per-metric deltas, config deltas, and per-case regression drill-down.
  - `crucible export <app> [--format csv|json] [--output FILE] [--tag ...]`.
  - `crucible config` (`save`/`list`/`show`/`rm`/`default`) — manage named configs; `crucible new app <name>` — scaffold; `crucible doctor [--network]` — health checks; `crucible dataset stats <app>` — dataset statistics.
- Named configs as versionable JSON files (`configs/<app>/<name>.json`), referenced by name via `--config` (or passed inline as JSON); a per-app default pointer (`configs/<app>/.default`) is used when `--config` is omitted.
- Model as an orthogonal axis: `--model` overrides the app model for a run while the LLM judge always uses the configured judge model; `--models a,b,c` records one run per model; `model_name` is stored per run so cross-model comparisons are possible.
- Run tags recorded at creation (`--tags a,b`) and `--tag` filtering on `show`/`compare`/`export`.
- **Optional DSPy integration** (install with `uv sync --group dspy`): each app can declare a `dspy_factory` on its `AppRegistration` returning a `DspyProgramSpec` (three callables: `build`, `prepare_example`, `prediction_to_output`); `crucible compile` runs `BootstrapFewShot` against the app's dataset and registered metrics; compile artifacts are JSON files loadable at evaluate time via `--program`.
- Configuration via environment / `.env` (`CRUCIBLE_*` prefix, `OPENAI_API_KEY` fallback, optional OpenAI-compatible `base_url` for local gateways).
- Versioned JSON datasets (`datasets/*_v1.json`) with a wrapper format (`{"version", "cases"}`), optional `corpus` for retrieval apps.
- Hermetic testing with `StubLLMClient`; no live API calls in the test suite.

## Out of scope

- **Serving**: no HTTP server, no API surface for running apps; Crucible is a CLI/experimentation tool, not a deployment platform.
- **Retrieval infrastructure**: no embeddings, vector database, index persistence, or chunking — retrieval is an in-memory keyword/substring scorer over a fixed corpus.
- **Scale**: no parallel/distributed trial execution, no multi-machine lineage; a single SQLite file is the whole store.
- **Retroactive re-tagging**: run tags are fixed at creation time; there is no command to re-tag an existing run.
- **Model-in-config**: config files never hold a model name — the model is an orthogonal CLI axis (`--model` / `--models`).
- **Regression alerting**: `compare --cases` reports per-case deltas (broke / fixed / unchanged) but there is no threshold-based alerting or CI integration.
- **Metric authoring beyond code**: metrics are Python classes; apps register their metric set, weights, search space, and default config via `register_app` (`registry.py`); no plugin loading or DSL.
- **Observability tooling**: no dashboards, charts, or web UI — lineage read-back is provided by the `crucible show/compare/export` commands (terminal tables and CSV/JSON file export), and raw `sqlite3` queries (an example query is in the README) remain available.
- **Production-grade robustness**: retry (tenacity) and structured-output repair exist, but there is no rate-limit budgeting, caching, or cost management for long optimize runs.
