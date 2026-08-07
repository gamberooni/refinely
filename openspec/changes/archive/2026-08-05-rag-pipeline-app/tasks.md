## 1. Dataset

- [x] 1.1 Create `datasets/rag_v1.json`: versioned wrapper with a 12-15 snippet corpus (including near-miss distractors) and 10 cases with `source_indices` referencing corpus positions
- [x] 1.2 Add loader tests: rag dataset parses to 10 cases, corpus loads, every case's `source_indices` are valid corpus indices

## 2. Retrieval

- [x] 2.1 Add `retrieve_snippets_indexed(question, corpus, top_k, strategy)` to `apps/retrieval.py` returning `list[tuple[int, str]]`; `keyword` strategy scores by keyword only, `hybrid` keeps the combined keyword+substring scorer
- [x] 2.2 Refactor `retrieve_snippets` to delegate to the indexed variant (hybrid strategy) without changing behavior
- [x] 2.3 Add tests: indexed variant returns correct corpus positions, strategy switch changes scoring, existing retrieval behavior unchanged

## 3. Metrics

- [x] 3.1 Add `RetrievalRecallMetric` to `eval/metrics.py`: |expected.source_indices ∩ output.retrieved_indices| / |expected.source_indices|, 0.0 when none retrieved
- [x] 3.2 Add `CitationAccuracyMetric`: |expected.source_indices ∩ output.cited_indices| / |cited_indices|, 0.0 when no citations
- [x] 3.3 Add tests for both metrics: full hit, partial, no overlap, empty citations, tolerance for dict outputs with extra keys
- [x] 3.4 Add `WEIGHT_SCHEMES["rag"]` (fuzzy 0.20, llm_judge 0.20, retrieval_recall 0.25, citation_accuracy 0.10, latency 0.10, cost 0.15) and extend the weight-sum test

## 4. RAG App

- [x] 4.1 Create `apps/rag.py` with `RAGAnswer` (answer, cited_snippets) and `RAGApp` implementing `ApplicationAdapter`; stages `_expand_query`, `_retrieve`, `_rerank`, `_generate` in one `asyncio.run` per case
- [x] 4.2 Add `build_metrics("rag")` registration: fuzzy, llm_judge, retrieval_recall, citation_accuracy, latency, cost
- [x] 4.3 Add app tests: default path single call, expansion toggle adds a `chat_text` call and uses the rewritten question, rerank toggle scores and reorders candidates, citations and retrieved indices in output, unknown variant raises without LLM calls, token/latency aggregation across calls

## 5. Search Space

- [x] 5.1 Add `sample_rag_config` to `optimize/search_space.py` with all six parameters (float, categorical, int, boolean)
- [x] 5.2 Add `SAMPLE_CONFIGS["rag"]` and `DEFAULT_CONFIGS["rag"]` (temperature 0.0, strict, hybrid, top_k 3, query_expansion off, rerank off)
- [x] 5.3 Add search space tests: rag keys and ranges via `FixedTrial`, dispatch works, default config keys match search space keys

## 6. CLI

- [x] 6.1 Register `rag` in `cli.py`: `APP_NAMES`, `DATASET_PATHS`, `_build_app` branch using `load_corpus`
- [x] 6.2 CLI smoke: `refinely --help` and `evaluate --help` show `rag` in the Choice
- [x] 6.3 Add an optimize smoke test (stub client) that runs the rag objective for 2 trials and records lineage rows

## 7. Docs

- [x] 7.1 Update README.md: rag app entry, weight scheme, usage examples
- [x] 7.2 Update AGENTS.md: add rag to commands and app registry notes
- [x] 7.3 Update CONTRIBUTING.md: apps, metrics, and test coverage lists

## 8. Verification

- [x] 8.1 Full pytest suite passes (no live API calls)
- [x] 8.2 Manual: `refinely evaluate rag` and `refinely optimize rag --trials 3` against the real gateway; lineage rows recorded and sensible
- [x] 8.3 Sync delta specs to main specs and archive the change
