# Crucible

Evaluate and optimize LLM application configurations. Crucible runs a configurable application over a dataset, scores it with weighted metrics, records full run lineage in SQLite, and uses Optuna (TPE) to search for better configurations.

Built as a greenfield prototype: three toy apps (extraction, QA, RAG), real LLM calls, spec-driven via OpenSpec.

## Quickstart

```bash
uv sync --group dev

# configure (copy and fill in)
cp .env.example .env
```

Configuration loads from environment variables prefixed `CRUCIBLE_` (or `.env` in the repo root), with an `OPENAI_API_KEY` fallback:

| Variable | Default | Description |
|---|---|---|
| `CRUCIBLE_OPENAI_API_KEY` | `OPENAI_API_KEY` | API key for the LLM provider |
| `CRUCIBLE_BASE_URL` | — | Base URL (OpenAI-compatible gateway, e.g. `http://localhost:4000/v1`) |
| `CRUCIBLE_MODEL_NAME` | `deepseek-v4-flash` | Model for app calls and LLM judging |
| `CRUCIBLE_LINEAGE_DB_PATH` | `lineage.db` | SQLite database for lineage + Optuna trials |

## Usage

```bash
# baseline evaluation of one app
uv run crucible evaluate extraction
uv run crucible evaluate qa
uv run crucible evaluate rag

# optimize a configuration (15 trials by default)
uv run crucible optimize extraction --trials 15
uv run crucible optimize qa --trials 3
uv run crucible optimize rag --trials 3

# compile a DSPy program (optional; requires uv sync --group dspy)
uv sync --group dspy
uv run crucible compile extraction --max-examples 20
uv run crucible evaluate extraction --program optimized_program.json
```

Or via the Makefile: `make install`, `make test`, `make evaluate APP=qa`, `make optimize APP=extraction TRIALS=15`, `make clean`.

The lineage database is plain SQLite — query it directly:

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
3. Every run — baseline or trial — is recorded to `evaluation_runs` / `metric_results` / `case_results`, linked by `optuna_trial_number`.

## Layout

```
apps/          demo apps: ExtractionApp, QAApp, RAGApp (sibling of src/, outside the crucible package)
src/crucible/
  core/        exceptions, settings
  llm/         AsyncOpenAIClient (JSON-schema-forced structured output, retries), TokenUsage
  retrieval.py generic keyword/hybrid retrieval (in-memory corpus)
  eval/        EvalCase + dataset loaders, generic metrics (fuzzy match, LLM judge, latency, cost), EvaluationRunner
  optimize/    Optuna objective + study runner (search spaces registered per app)
  tracking/    LineageDB (SQLite schema + record/query helpers; dspy_compiles table for compile lineage)
  registry.py  app registry: apps self-register metrics, weights, search space, defaults
  dspy/        optional DSPy compile harness (DspyProgramSpec, bridge, compile_program, configure_lm)
tests/         pytest suite — stub LLM client, zero live API calls
```

## Testing

```bash
uv run pytest tests/ -q   # 119 tests, no network
```

## Using crucible in your codebase

Crucible is a framework, not a monolith: your app plugs in as a plain object with `execute(input, config) -> Result` (duck-typed, no protocol class), and everything else (metrics, search space, defaults, weights) is either registered per app or passed explicitly. Two styles:

- **Registry app** — any module (in the `apps/` directory or in your own repo) calls `register_app` at import and is declared as an entry point in group `crucible.apps`; the app appears in `crucible evaluate` / `crucible optimize` automatically.
- **Library driver** — depend on crucible from your own repository and drive `build_objective` + `run_study` directly, passing your adapter, metrics, search space, and weights.

See [Integration](docs/integration.md) for a full guide with example code.

## Documentation

- [Overview](docs/overview.md) — what Crucible is, purpose, goals, in-scope and out-of-scope.
- [Architecture](docs/architecture.md) — system concepts, component diagram, data flow diagrams, and sequence diagrams for evaluation, optimization, and the RAG pipeline.
- [Integration](docs/integration.md) — how to use crucible in your own codebase: registry apps vs. library driver.
