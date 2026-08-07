## Why

The `docs/` folder specifies a full enterprise "AI Application Evaluation & Optimization Platform" (evaluation engine + metrics + DSPy + Optuna + MLflow + multi-tenant governance). Before investing in any of that, we need to validate the *core idea* cheaply: that a black-box `ApplicationAdapter.execute(input, config)` contract can decouple genuinely different toy applications from a shared evaluation harness, and that wrapping evaluation as an Optuna objective function actually improves an aggregate score over trials — with every result traceable back to the application, dataset, and configuration that produced it (lineage). A minimal, single-repo prototype answers this in days instead of committing to the full platform architecture.

## What Changes

- Add two toy GenAI applications behind a shared `ApplicationAdapter` protocol: a structured-extraction app and an in-memory retrieval-lite Q&A app — both making real OpenAI calls via an adapted `AsyncOpenAIClient`.
- Add a minimal evaluation engine: JSON-file datasets, an `EvaluationRunner` loop, a small set of metrics (exact-match, fuzzy-match, LLM-judge, latency, cost), and weighted score aggregation.
- Add a minimal optimization engine: per-app Optuna search spaces, a single-objective `study.optimize` loop (TPE sampler, 15 trials/app) that treats "run the evaluation" as the objective function.
- Add SQLite-backed experiment lineage tracking (shared DB file with Optuna's own storage) recording `evaluation_runs`, `metric_results`, and `case_results` so every run's app/dataset-version/config/score is queryable and comparable.
- Add a `click`-based CLI (`refinely`) to run evaluations and optimizations per app.
- Add `pydantic-settings`-based configuration and supporting `pytest` unit tests (stubbed LLM client, no live-API tests).

## Capabilities

### New Capabilities
- `application-adapter`: The black-box `ApplicationAdapter` protocol and the two toy applications (extraction, QA) that implement it, each with their own config schema and LLM-backed execution.
- `evaluation-engine`: Dataset loading/versioning, the `EvaluationRunner` execution loop, the metric plugins (exact-match, fuzzy-match, LLM-judge, latency, cost), and weighted aggregation into a single score.
- `optimization-engine`: Per-app Optuna search-space definitions, objective-function construction that wraps the evaluation engine, and the single-objective TPE optimization loop.
- `experiment-lineage-tracking`: SQLite schema and read/write operations for `evaluation_runs`, `metric_results`, and `case_results`, giving reproducible lineage (app + dataset version + configuration + metrics) per run and per case.
- `cli`: The `refinely` command-line entrypoint for running evaluations and optimizations against either toy application.

### Modified Capabilities
_(none — this is a greenfield prototype; no existing specs in this repo)_

## Impact

- **New code**: `src/refinely/core/` (settings, exceptions), `src/refinely/llm/` (client, usage), `src/refinely/apps/` (protocol, extraction, qa), `src/refinely/datasets/` (loader + JSON fixtures), `src/refinely/eval/` (runner, metrics), `src/refinely/optimize/` (search space, objective), `src/refinely/tracking/` (SQLite db), `src/refinely/cli.py`.
- **New dependencies**: `openai`, `pydantic>=2`, `pydantic-settings`, `tenacity`, `optuna`, `click`; dev dependency `pytest`.
- **New runtime dependency**: requires `OPENAI_API_KEY` in the environment to run real evaluations/optimizations (unit tests use a stub `LLMClient` and require no network access).
- **Storage**: a local SQLite file (shared between Optuna's internal trial storage and the custom lineage tables) — no external services, no MLflow, no DSPy, no queue/worker infrastructure.
- **No production impact**: this is a standalone prototype in `src/refinely/`; it does not touch or depend on the enterprise platform described in `docs/`.
