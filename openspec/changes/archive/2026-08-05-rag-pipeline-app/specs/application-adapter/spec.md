## ADDED Requirements

### Requirement: RAG application adapter
The system SHALL provide a `RAGApp` that implements `ApplicationAdapter`, running a multi-stage retrieval-augmented QA pipeline (query expansion, retrieval, reranking, generation with citations) with up to four LLM calls per case.

#### Scenario: Adapter shape stays uniform
- **WHEN** the evaluation engine runs `RAGApp.execute(input, config)` through the same `EvaluationRunner` used for `ExtractionApp` and `QAApp`
- **THEN** no application-specific branching SHALL be required inside the evaluation engine

#### Scenario: Sync facade over async pipeline
- **WHEN** `RAGApp.execute` is invoked with a config that enables expansion and reranking
- **THEN** all LLM calls for the case SHALL run inside a single `asyncio.run` and return a sync `Result` with aggregated token usage and total latency
