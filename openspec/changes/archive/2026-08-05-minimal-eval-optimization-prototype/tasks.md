## 1. Project Setup

- [x] 1.1 Add dependencies to `pyproject.toml`: `openai`, `pydantic>=2`, `pydantic-settings`, `tenacity`, `optuna`, `click`; add `pytest` under a dev dependency group
- [x] 1.2 Add `[project.scripts]` entry `crucible = "crucible.cli:main"` to `pyproject.toml`
- [x] 1.3 Create package skeleton: `src/crucible/{core,llm,apps,datasets,eval,optimize,tracking}/__init__.py`
- [x] 1.4 Install dependencies and verify `python -c "import crucible"` succeeds

## 2. Core: Settings and Exceptions

- [x] 2.1 Implement `crucible/core/exceptions.py` with `LLMError` and `EvalError` exception classes
- [x] 2.2 Implement `crucible/core/settings.py` using `pydantic-settings` `BaseSettings` with fields for OpenAI API key, model name, and lineage DB path, loading from environment / `.env`
- [x] 2.3 Unit test: settings load correctly from environment variables (using monkeypatched env, no real API key required)

## 3. LLM Client

- [x] 3.1 Implement `crucible/llm/usage.py` with the `TokenUsage` model (`prompt_tokens`, `completion_tokens`)
- [x] 3.2 Implement `crucible/llm/client.py`: adapt the `AsyncOpenAIClient` pattern (rename imports to `crucible`), including `LLMClient` Protocol, `chat_structured` (JSON-schema-forced output, fence-stripping, repair-retry), `chat_text`, `tenacity` retry wrapping, and per-model temperature support detection
- [x] 3.3 Implement a stub `LLMClient` test double (canned `chat_structured`/`chat_text` responses) for use in unit tests
- [x] 3.4 Unit test: `_strip_json_fences` and `_extract_json_from_prose` helpers against fenced/unfenced/prose-wrapped JSON strings
- [x] 3.5 Unit test: stub client satisfies the `LLMClient` Protocol and returns expected canned `TokenUsage`

## 4. Application Adapter Protocol

- [x] 4.1 Implement `crucible/apps/protocol.py`: `ApplicationAdapter` Protocol with `execute(input: dict, config: dict) -> Result`, and a shared `Result` model (output + execution metadata including token usage and latency)

## 5. Extraction Application

- [x] 5.1 Implement `crucible/apps/extraction.py`: `ExtractionApp` implementing `ApplicationAdapter`, using `chat_structured` with a Pydantic extraction response model, `temperature` and `system_prompt_variant` config knobs (two hand-written prompt templates: "strict"/"verbose")
- [x] 5.2 Unit test: `ExtractionApp.execute` against the stub `LLMClient` returns a `Result` with the expected structured output shape and metadata

## 6. Retrieval-lite QA Application

- [x] 6.1 Implement in-memory keyword/substring snippet retrieval function (given a question and a corpus, return up to `top_k` matching snippets)
- [x] 6.2 Implement `crucible/apps/qa.py`: `QAApp` implementing `ApplicationAdapter`, retrieving snippets then calling `chat_text`/`chat_structured`, with `temperature`, `top_k`, and `system_prompt_variant` config knobs
- [x] 6.3 Unit test: snippet retrieval returns the correct number and relevance of snippets for known inputs
- [x] 6.4 Unit test: `QAApp.execute` against the stub `LLMClient` returns a `Result` with the expected answer shape and metadata

## 7. Datasets

- [x] 7.1 Define `EvalCase` model (`id`, `input`, `expected`) in `crucible/datasets/loader.py`
- [x] 7.2 Implement dataset loader: parse a JSON file into `list[EvalCase]`, raising a clear error on missing required fields
- [x] 7.3 Author `datasets/extraction_v1.json` with 8-12 cases (text input + expected structured field, e.g. sentiment or invoice total)
- [x] 7.4 Author `datasets/qa_v1.json` with 8-12 cases (question input + expected answer, plus an in-memory corpus of snippets to retrieve from)
- [x] 7.5 Unit test: loader parses both dataset files without error and produces the expected case count

## 8. Evaluation Metrics

- [x] 8.1 Implement `crucible/eval/metrics.py`: `Metric` base/protocol with `evaluate(case, output) -> MetricResult`
- [x] 8.2 Implement `ExactMatchMetric` (field-level equality, numeric tolerance where applicable)
- [x] 8.3 Implement `FuzzyMatchMetric` (normalized substring/token-overlap match)
- [x] 8.4 Implement `LatencyMetric` (wall-clock duration per case, from `Result` metadata)
- [x] 8.5 Implement `CostMetric` (estimated cost derived from `TokenUsage` in `Result` metadata)
- [x] 8.6 Implement `LLMJudgeMetric` (1-5 relevance/faithfulness score via LLM judge call using the `LLMClient`)
- [x] 8.7 Implement weighted aggregation function: combine per-case metric scores into one `aggregate_score`, with distinct weight schemes for extraction (ExactMatch/Latency/Cost) and QA (FuzzyMatch/LLMJudge/Latency/Cost)
- [x] 8.8 Unit test: each metric's `evaluate` against known input/output/expected triples produces the expected score
- [x] 8.9 Unit test: weighted aggregation produces the expected `aggregate_score` for a known set of per-case scores and weights

## 9. Evaluation Runner

- [x] 9.1 Implement `crucible/eval/runner.py`: `EvaluationRunner.run(dataset, app, config)` executing every case through `app.execute`, collecting per-case output/expected/score, tolerating individual case failures without aborting the run
- [x] 9.2 Unit test: `EvaluationRunner.run` against a stub `ApplicationAdapter` and a small in-memory dataset produces one result record per case and a correct `aggregate_score`
- [x] 9.3 Unit test: a single case raising an exception during metric computation does not abort the overall run

## 10. Experiment Lineage Tracking

- [x] 10.1 Implement `crucible/tracking/db.py`: SQLite schema initialization for `evaluation_runs`, `metric_results`, `case_results` (idempotent — safe to call against an existing DB file, including one already containing Optuna's internal tables)
- [x] 10.2 Implement `record_run(...)`: insert one `evaluation_runs` row, N `metric_results` rows, and N `case_results` rows for a completed `EvaluationRunner` result, generating a unique `run_id`
- [x] 10.3 Implement query helpers: fetch best-scoring run per `app_name`, fetch `case_results` for a given `run_id` ordered by score
- [x] 10.4 Unit test: schema initializes cleanly on a fresh file and is a no-op on a file that already has the tables
- [x] 10.5 Unit test: `record_run` followed by the query helpers round-trips correctly (JSON fields parse back to the original dicts)

## 11. Optimization Engine

- [x] 11.1 Implement `crucible/optimize/search_space.py`: per-app Optuna search-space sampling functions (extraction: `temperature`, `system_prompt_variant`; QA: `temperature`, `top_k`, `system_prompt_variant`)
- [x] 11.2 Implement `crucible/optimize/objective.py`: `build_objective(app, dataset, lineage_db_path)` returning an Optuna objective function that samples a config, runs `EvaluationRunner`, records the run to lineage with `optuna_trial_number` set, and returns `aggregate_score`
- [x] 11.3 Implement study creation/execution: `optuna.create_study(direction="maximize", sampler=TPESampler(), storage="sqlite:///<lineage-db-path>")` and `study.optimize(objective, n_trials=15)` (trial count overridable)
- [x] 11.4 Unit test: search-space sampling functions produce configs with only the expected keys and value ranges for each app
- [x] 11.5 Unit test: objective function, run against a stub `ApplicationAdapter`/`LLMClient`, returns a float and records a lineage row with the correct `optuna_trial_number`

## 12. CLI

- [x] 12.1 Implement `crucible/cli.py` with `click` group `main` and two subcommands: `evaluate <app>` and `optimize <app> [--trials N]` (app is `extraction` or `qa`)
- [x] 12.2 `evaluate` subcommand: runs the app's baseline `EvaluationRunner`, prints `aggregate_score`, records the run to lineage
- [x] 12.3 `optimize` subcommand: runs the app's Optuna study for the given trial count (default 15), prints the best trial's score and configuration, records every trial to lineage
- [x] 12.4 Manual verification: run `crucible evaluate extraction`, `crucible evaluate qa`, `crucible optimize extraction --trials 3`, `crucible optimize qa --trials 3` against the real OpenAI API and confirm sensible output and lineage rows are written

## 13. Final Verification

- [x] 13.1 Run the full unit test suite (`pytest`) and confirm all tests pass with zero live API calls
- [x] 13.2 Run `crucible optimize extraction --trials 15` and `crucible optimize qa --trials 15` end-to-end; confirm `aggregate_score` trends upward (or at least varies meaningfully) across trials via a lineage query
- [x] 13.3 Query the lineage DB directly (e.g. via `sqlite3` CLI) to confirm `evaluation_runs`, `metric_results`, and `case_results` are populated and joinable as designed
