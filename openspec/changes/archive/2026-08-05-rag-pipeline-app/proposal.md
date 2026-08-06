## Why

The two existing toy apps (extraction, QA) each make a single LLM call with 2-3 config parameters. They do not exercise the framework's deeper seams: multi-stage pipelines, multiple LLM calls per case, conditional stages driven by config toggles, and dataset-aware metrics. A richer app is needed to stress test the framework (search space sampling, metric plugins, weight aggregation, lineage recording, stub-based testing) before it grows further.

## What Changes

- Add a new `rag` app: a retrieval-augmented QA pipeline with four stages (query expansion, retrieval, reranking, generation) and up to four LLM calls per case.
- Add six config parameters to the search space: `temperature` (float), `system_prompt_variant` (categorical), `retrieval_strategy` (categorical keyword/hybrid), `top_k` (int), `query_expansion` (bool), `rerank` (bool).
- Add two deterministic retrieval metrics: `retrieval_recall` (expected source snippets vs retrieved) and `citation_accuracy` (cited vs expected sources, precision).
- Add a new versioned dataset `datasets/rag_v1.json` with a corpus and cases whose expected answers carry 0-based source indices.
- Extend retrieval with a strategy switch and an index-aware variant returning matched corpus positions.
- Register the app end-to-end: metrics (`build_metrics`), weight scheme, search space, default config, CLI app names/dataset paths.
- Update docs (README, AGENTS.md, CONTRIBUTING.md).

## Capabilities

### New Capabilities
- `rag-app`: the multi-stage RAG pipeline application (components, config contract, citation output)

### Modified Capabilities
- `application-adapter`: extend to cover the new RAG application's adapter contract
- `evaluation-engine`: add retrieval-based metric requirements and a new weight scheme
- `optimization-engine`: add a requirement for per-app search spaces covering mixed parameter types (float/categorical/int/bool)
- `cli`: add the new app to the CLI's app registry

## Impact

- New: `src/crucible/apps/rag.py`, `datasets/rag_v1.json`, tests for the app/metrics/search space
- Modified: `src/crucible/apps/retrieval.py` (strategy + indexed variant), `src/crucible/eval/metrics.py` (2 metrics, weight scheme, `build_metrics`), `src/crucible/optimize/search_space.py`, `src/crucible/cli.py`, README.md, AGENTS.md, CONTRIBUTING.md
- No changes to settings, lineage schema, or the evaluation runner
- Full test suite grows from 65 to roughly 75 tests; no new dependencies
