## Context

Crucible currently ships two toy apps behind the `ApplicationAdapter` protocol: `extraction` (one `chat_structured` call) and `qa` (deterministic retrieval + one `chat_structured` call). Each app is registered in five places: `build_metrics` + `WEIGHT_SCHEMES` (eval/metrics.py), `SAMPLE_CONFIGS` + `DEFAULT_CONFIGS` (optimize/search_space.py), `APP_NAMES` + `DATASET_PATHS` + `_build_app` (cli.py), and a versioned dataset under `datasets/`. Apps are sync (`execute`) and wrap the async `AsyncOpenAIClient` via `asyncio.run`. Metrics are either deterministic or judge-based; all scores are clamped to 0..1 and weighted per scheme summing to 1.0. Case outputs are JSON-serializable dicts recorded per case in lineage.

The framework has no app with multiple LLM calls per case, conditional pipeline stages, mixed-type search spaces (float/categorical/int/bool), or metrics that read the retrieval process rather than just the final answer. This change adds one such app to validate those seams.

## Goals / Non-Goals

**Goals:**
- Add a `rag` app: query expansion (optional) -> retrieval -> reranking (optional) -> generation with citations
- Up to four LLM calls per case, all serialized in one event loop
- Six config parameters across all Optuna types (float, categorical, int, bool)
- Two deterministic retrieval metrics (`retrieval_recall`, `citation_accuracy`)
- Full registration: metrics, weights, search space, defaults, CLI, dataset
- Keep the public sync `execute` API and the versioned-dataset format unchanged

**Non-Goals:**
- No new dependencies, no schema changes to lineage, no changes to `EvaluationRunner`
- No chunking/embedding-based retrieval (in-memory scoring stays)
- No change to the existing `qa`/`extraction` apps beyond the shared retrieval helper

## Decisions

### 1. Four-stage pipeline with optional stages driven by config toggles
Stages: `_expand_query` (LLM `chat_text`, only when `query_expansion=True`), `_retrieve` (deterministic), `_rerank` (LLM `chat_structured`, only when `rerank=True` and >1 candidate), `_generate` (LLM `chat_structured`, always).
Rationale: toggles create conditional call graphs - the stub must consume a variable number of canned responses per case, which is exactly the stress this app exists to test. Alternative considered: unconditional stages with fixed call counts - rejected, weaker stress.

### 2. One `asyncio.run` per case around an internal async pipeline
The app's public `execute` stays sync; the whole pipeline runs inside a single `asyncio.run(self._execute_async(...))`. Existing apps create one event loop per LLM call; a 4-call pipeline would create four per case. One loop per case is cheaper and keeps the sync-over-async seam intact.
Rationale: fewer event-loop spins, one latency measurement span.

### 3. Index-aware retrieval with a strategy switch
`apps/retrieval.py` gains `retrieve_snippets_indexed(question, corpus, top_k, strategy) -> list[tuple[int, str]]` where the int is the 0-based corpus position. `strategy` selects keyword-only scoring or the existing hybrid (keyword + substring) scorer. `retrieve_snippets` keeps its signature (delegates to the indexed variant with hybrid strategy) so `qa` is untouched.
Rationale: corpus snippets have no ids; 0-based indices are the identity (matches the dataset's `source_indices`).

### 4. New metrics are deterministic and answer-independent
`retrieval_recall` = |expected.source_indices intersect output.retrieved_indices| / |expected.source_indices| (0.0 if expected empty/none retrieved). `citation_accuracy` = |expected.source_indices intersect output.cited_indices| / |cited_indices| (precision; 0.0 if no citations).
Rationale: no extra LLM calls; they measure the retrieval stage directly, which `fuzzy_match`/`llm_judge` cannot see. Precision chosen over recall to penalize hallucinated citations.

### 5. Weight scheme for six metrics sums to 1.0
`rag`: fuzzy_match 0.20, llm_judge 0.20, retrieval_recall 0.25, citation_accuracy 0.10, latency 0.10, cost 0.15.
Rationale: answer quality dominates, retrieval matters, cost/latency get the same weight as other apps. `aggregate_scores` treats a missing metric as 0.0, so every metric must be present in `build_metrics` output.

### 6. Default config is the cheap, deterministic path
`DEFAULT_CONFIGS["rag"]` = temperature 0.0, strict, hybrid, top_k 3, query_expansion False, rerank False - the same shape as the other apps' baselines (deterministic retrieval + one call) so `evaluate rag` is comparable.

## Risks / Trade-offs

- **Stub queue accounting complexity** -> tests build the canned-response queue per call count; helper fixture in tests counts calls per case (expansion/rerank/generate/judge).
- **Rerank with a single candidate** -> rerank stage skips the LLM call when <=1 candidate to avoid a pointless call; documented in tests.
- **citation_accuracy precision is strict** -> extra unrelated citations penalize the score; intended behavior (hallucination detection), flagged in docs.
- **`keyword` strategy can return 0 snippets** -> recall = 0.0, answer path still works (no snippets in prompt); same behavior as `qa` today.
- **Real-API latency** -> up to 4 serial LLM calls per case makes 15-trial optimize runs slower (~4x qa); manual verification uses --trials 3 smoke first.
