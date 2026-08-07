# Refinely

Evaluate and optimize LLM application configurations. Refinely runs a configurable application over a dataset, scores it with weighted metrics, records full run lineage in SQLite, and uses Optuna (TPE) to search for better configurations.

Built as a greenfield prototype: three toy apps (extraction, QA, RAG), real LLM calls, spec-driven via OpenSpec.

## Quickstart

```bash
uv sync --group dev

# configure (copy and fill in)
cp .env.example .env
```

Configuration loads from environment variables prefixed `REFINELY_` (or `.env` in the repo root), with an `OPENAI_API_KEY` fallback:

| Variable | Default | Description |
|---|---|---|
| `REFINELY_OPENAI_API_KEY` | `OPENAI_API_KEY` | API key for the LLM provider |
| `REFINELY_BASE_URL` | — | Base URL (OpenAI-compatible gateway, e.g. `http://localhost:4000/v1`) |
| `REFINELY_MODEL_NAME` | `deepseek-v4-flash` | Default model for app calls and the LLM judge |
| `REFINELY_LINEAGE_DB_PATH` | `lineage.db` | SQLite database for lineage + Optuna trials |

## Usage

```bash
# baseline evaluation of one app
uv run refinely evaluate extraction
uv run refinely evaluate qa
uv run refinely evaluate rag

# evaluate with a stored config or an ad-hoc one (both merged over the app's defaults)
uv run refinely evaluate extraction --config my-run
uv run refinely evaluate extraction --config '{"temperature": 0.4, "system_prompt_variant": "verbose"}'

# named configs live at configs/<app>/<name>.json, managed via the config group
uv run refinely config save my-run --app extraction --config '{"temperature": 0.4}'
uv run refinely config list
uv run refinely config default extraction --set my-run   # no --config → this default

# the model is an orthogonal axis, not part of a config file
uv run refinely evaluate extraction --model deepseek-v4-flash
uv run refinely evaluate extraction --models a,b,c        # one recorded run per model
uv run refinely evaluate extraction --tags candidate,prod # tag a run for later filtering

# optimize a configuration (15 trials by default; best config auto-saved to configs/<app>/opt-best.json)
uv run refinely optimize extraction --trials 15
uv run refinely optimize qa --trials 3
uv run refinely optimize rag --trials 3

# compile a DSPy program (optional; requires uv sync --group dspy)
uv sync --group dspy
uv run refinely compile extraction --max-examples 20
uv run refinely evaluate extraction --program optimized_program.json
```

Read results back from the lineage database without writing any SQL:

```bash
# run history, newest first, with best-run/best-compile summaries
uv run refinely show extraction
uv run refinely show extraction --tag candidate      # filter by tag
# per-case results for one run (worst cases first; run ids may be prefix-abbreviated)
uv run refinely show extraction --run 3f2a9c1d       # renders per-case errors + errored count
# per-metric deltas between runs, vs. the previous run or an explicit baseline
uv run refinely compare extraction
uv run refinely compare extraction --baseline 3f2a9c1d
uv run refinely compare extraction --diff-config      # config delta vs. the baseline run
uv run refinely compare extraction --cases            # per-case broke/fixed/unchanged drilldown
uv run refinely compare extraction --model gpt-4o     # restrict to one model
# export runs to CSV or JSON
uv run refinely export extraction --format csv --output extraction_runs.csv
uv run refinely export extraction --tag prod
```

Or via the Makefile: `make install`, `make test`, `make evaluate APP=qa`, `make optimize APP=extraction TRIALS=15`, `make clean`.

Developer tooling for building and debugging apps:

```bash
# scaffold a new app (apps/<name>.py + datasets/<name>_v1.json; prints the pyproject entry point to add)
uv run refinely new app myapp
# health checks (app discovery, datasets, schema, env key); --network also probes the gateway
uv run refinely doctor
# dataset statistics + malformed-case report
uv run refinely dataset stats extraction
```

The lineage database is plain SQLite — query it directly too:

```bash
sqlite3 lineage.db "SELECT app_name, optuna_trial_number, aggregate_score FROM evaluation_runs ORDER BY created_at;"
```

## Apps and datasets

- **Extraction** (`extraction_v1.json`) — extracts a named field (sentiment) from text; config: `temperature`, `system_prompt_variant` (`strict`/`verbose`).
- **QA** (`qa_v1.json`) — retrieval-lite question answering over a fixed corpus; config: `temperature`, `system_prompt_variant`, `top_k` (1–5).
- **RAG** (`rag_v1.json`) — 4-stage pipeline: query expansion (LLM, optional) → deterministic retrieval → reranking (LLM, optional) → answer generation (LLM); config: `temperature`, `system_prompt_variant`, `retrieval_strategy` (`keyword`/`hybrid`), `top_k` (1–6), `query_expansion`, `rerank`. Output includes `retrieved_indices` and `cited_indices` for retrieval/citation scoring.

Datasets use a versioned wrapper format: `{"version": "...", "cases": [...]}` (QA and RAG also carry a `corpus`; RAG cases expect `source_indices` in their expected answers).

## How evaluation works

1. `app.execute(input, config)` (any object with that method) returns `Result` (output, token usage, latency).
2. Metrics score each case (missing/erroring cases score 0, never aborting the run):
   - extraction: `exact_match` 0.7, `latency` 0.15, `cost` 0.15
   - qa: `fuzzy_match` 0.4, `llm_judge` 0.3, `latency` 0.15, `cost` 0.15
   - rag: `fuzzy_match` 0.2, `llm_judge` 0.2, `retrieval_recall` 0.25, `citation_accuracy` 0.1, `latency` 0.1, `cost` 0.15
3. Every run — baseline or trial — is recorded to `evaluation_runs` / `metric_results` / `case_results`, linked by `optuna_trial_number`. Runs carry the `model_name` and any `tags` given at creation; per-case errors are persisted on `case_results.error` (missing/erroring cases score 0, never aborting the run).

## Layout

```
apps/          demo apps: ExtractionApp, QAApp, RAGApp (sibling of src/, outside the refinely package)
apps/common.py generic keyword/hybrid retrieval (in-memory corpus), shared by QA and RAG
src/refinely/
  core/        exceptions, settings
  llm/         AsyncOpenAIClient (JSON-schema-forced structured output, retries), TokenUsage
  config.py    named configs: configs/<app>/<name>.json + .default pointer (config CLI group)
  eval/        EvalCase + dataset loaders + dataset_stats, generic metrics (fuzzy match, LLM judge, latency, cost), EvaluationRunner
  optimize/    Optuna objective + study runner (search spaces registered per app)
  tracking/    LineageDB (SQLite schema + record/query helpers; dspy_compiles table for compile lineage)
  registry.py  app registry: apps self-register metrics, weights, search space, defaults
  dspy/        optional DSPy compile harness (DspyProgramSpec, bridge, compile_program, configure_lm)
  cli/         CLI package: main group (context.py call-time seam), evaluate, optimize, compile, config_cmds, readback, devtools
  devtools/    developer tooling: scaffold (new app), doctor (health checks)
configs/       named configs as JSON files: configs/<app>/<name>.json, opt-best.json, .default pointer
tests/         pytest suite — stub LLM client, zero live API calls
```

## Testing

```bash
uv run pytest tests/ -q
```

## Using refinely in your codebase

Refinely is a framework, not a monolith: your app plugs in as a plain object with `execute(input, config) -> Result` (duck-typed, no protocol class), and everything else (metrics, search space, defaults, weights) is either registered per app or passed explicitly. Two styles:

- **Registry app** — any module (in the `apps/` directory or in your own repo) calls `register_app` at import and is declared as an entry point in group `refinely.apps`; the app appears in `refinely evaluate` / `refinely optimize` automatically. `refinely new app <name>` scaffolds the module + dataset stub for you.
- **Library driver** — depend on refinely from your own repository and drive `build_objective` + `run_study` directly, passing your adapter, metrics, search space, and weights.

See [Integration](docs/integration.md) for a full guide with example code.

## Documentation

- [Overview](docs/overview.md) — what Refinely is, purpose, goals, in-scope and out-of-scope.
- [Architecture](docs/architecture.md) — system concepts, component diagram, data flow diagrams, and sequence diagrams for evaluation, optimization, and the RAG pipeline.
- [Integration](docs/integration.md) — how to use refinely in your own codebase: registry apps vs. library driver.
