# Application Adapter Specification

## Purpose

The black-box `ApplicationAdapter` protocol and the two toy applications (extraction, QA) that implement it, each with their own config schema and LLM-backed execution. The protocol decouples genuinely different application designs from the shared evaluation engine. (Adapted from change `minimal-eval-optimization-prototype`.)
## Requirements
### Requirement: ApplicationAdapter protocol
The system SHALL define an `ApplicationAdapter` protocol with a single method `execute(input: dict, config: dict) -> Result` that any toy application implements, so the evaluation engine never depends on application-specific internals.

#### Scenario: Adapter signature is shape-agnostic
- **WHEN** the evaluation engine calls `adapter.execute(input, config)` on any registered application
- **THEN** the call SHALL succeed using only the `input` dict and `config` dict, without the evaluation engine importing or referencing any application-specific class or module

### Requirement: Extraction application
The system SHALL provide an `ExtractionApp` that implements `ApplicationAdapter`, performing structured field extraction (e.g. sentiment label or invoice total) from free-text input via a real OpenAI call using `chat_structured`.

#### Scenario: Extraction produces a structured result
- **WHEN** `ExtractionApp.execute(input, config)` is called with a text input and a configuration containing `temperature` and `system_prompt_variant`
- **THEN** the app SHALL call the LLM client's `chat_structured` method with a Pydantic response model and return a `Result` containing the extracted structured output and execution metadata (including token usage)

### Requirement: Retrieval-lite QA application
The system SHALL provide a `QAApp` that implements `ApplicationAdapter`, answering questions using an in-memory keyword/substring snippet retrieval step (no vector database) followed by an LLM call.

#### Scenario: QA retrieves snippets then answers
- **WHEN** `QAApp.execute(input, config)` is called with a question and a configuration containing `temperature`, `top_k`, and `system_prompt_variant`
- **THEN** the app SHALL retrieve up to `top_k` in-memory snippets matching the question via keyword/substring search, inject them into the LLM prompt, and return a `Result` containing the answer text and execution metadata

### Requirement: Adapter shape independence
The two toy applications SHALL differ meaningfully in configuration shape and execution style (structured-output extraction vs. retrieval-augmented text answer), so the shared adapter interface is verified to work across genuinely different application designs.

#### Scenario: Same evaluation harness runs both apps unmodified
- **WHEN** the evaluation engine runs the same `EvaluationRunner` against both `ExtractionApp` and `QAApp` with their respective datasets and configs
- **THEN** no application-specific branching SHALL be required inside the evaluation engine

### Requirement: RAG application adapter
The system SHALL provide a `RAGApp` that implements `ApplicationAdapter`, running a multi-stage retrieval-augmented QA pipeline (query expansion, retrieval, reranking, generation with citations) with up to four LLM calls per case.

#### Scenario: Adapter shape stays uniform
- **WHEN** the evaluation engine runs `RAGApp.execute(input, config)` through the same `EvaluationRunner` used for `ExtractionApp` and `QAApp`
- **THEN** no application-specific branching SHALL be required inside the evaluation engine

#### Scenario: Sync facade over async pipeline
- **WHEN** `RAGApp.execute` is invoked with a config that enables expansion and reranking
- **THEN** all LLM calls for the case SHALL run inside a single `asyncio.run` and return a sync `Result` with aggregated token usage and total latency
