# rag-app Specification

## Purpose

The `RAGApp` reference application: a multi-stage retrieval-augmented QA pipeline (optional query expansion → deterministic snippet retrieval → optional LLM reranking → structured generation with citations) that stresses the framework with conditional stages, multiple LLM calls per case, dataset-aware retrieval metrics, and a mixed-type search space. (Adapted from change `rag-pipeline-app`.)
## Requirements
### Requirement: RAG application
The system SHALL provide a `RAGApp` that implements `ApplicationAdapter`, running a multi-stage retrieval-augmented QA pipeline: optional query expansion, deterministic snippet retrieval, optional LLM reranking, and structured generation with citations.

#### Scenario: Pipeline runs without optional stages
- **WHEN** `RAGApp.execute(input, config)` is called with `query_expansion=False` and `rerank=False`
- **THEN** the app SHALL retrieve up to `top_k` snippets from the corpus, generate an answer with `chat_structured`, and return a `Result` whose output contains the answer and the 0-based indices of both retrieved and cited snippets

#### Scenario: Optional stages add LLM calls
- **WHEN** `query_expansion=True` is set
- **THEN** the app SHALL first rewrite the question via a `chat_text` LLM call and use the rewritten question for retrieval
- **WHEN** `rerank=True` is set with more than one candidate
- **THEN** the app SHALL score the candidates via a `chat_structured` LLM call, reorder them, and keep the top `top_k`

#### Scenario: Config contract is validated
- **WHEN** `RAGApp.execute` is called with an unknown `system_prompt_variant`
- **THEN** the app SHALL raise `EvalError` without making any LLM calls

### Requirement: Pipeline execution semantics
The RAG pipeline SHALL execute all LLM calls for one case within a single event loop and report combined token usage and end-to-end latency.

#### Scenario: Token and latency aggregation
- **WHEN** a case triggers expansion, reranking, and generation calls
- **THEN** the returned `Result` SHALL sum prompt and completion tokens across all calls and set `latency_seconds` to the wall time of the whole pipeline

#### Scenario: Retrieval strategy switch
- **WHEN** `retrieval_strategy` is `keyword`
- **THEN** retrieval SHALL score snippets with keyword matching only
- **WHEN** `retrieval_strategy` is `hybrid`
- **THEN** retrieval SHALL use the combined keyword and substring scorer
