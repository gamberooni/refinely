# Evaluation Engine Specification

## Purpose

Dataset loading/versioning, the `EvaluationRunner` execution loop, the metric plugins (exact-match, fuzzy-match, LLM-judge, latency, cost), and weighted aggregation into a single score. (Adapted from change `minimal-eval-optimization-prototype`.)
## Requirements
### Requirement: Versioned JSON datasets
The system SHALL store evaluation datasets as versioned JSON files (e.g. `extraction_v1.json`, `qa_v1.json`), each containing a list of cases with an `id`, `input`, and `expected` field, and provide a loader that parses these into typed `EvalCase` objects.

#### Scenario: Loading a dataset file
- **WHEN** the dataset loader is given a path to a dataset JSON file
- **THEN** it SHALL return a list of `EvalCase` objects, each with `id`, `input`, and `expected` populated, and SHALL raise a clear error if any case is missing a required field

### Requirement: EvaluationRunner execution loop
The system SHALL provide an `EvaluationRunner` that, given a dataset, an `ApplicationAdapter`, and a configuration, executes every case through the adapter and collects each case's input, output, and expected value for scoring.

#### Scenario: Running an evaluation over a dataset
- **WHEN** `EvaluationRunner.run(dataset, app, config)` is called
- **THEN** the runner SHALL call `app.execute(case.input, config)` for every case in the dataset and produce a per-case record containing the case id, output, and expected value, without stopping the whole run if a single case's metric computation fails

### Requirement: Metric plugins
The system SHALL provide the following metric implementations, each computing a numeric score for a single case: `ExactMatchMetric` (field-level equality, used by the extraction app), `FuzzyMatchMetric` (normalized substring/token-overlap match, used by the QA app), `LatencyMetric` (wall-clock duration per case, shared), `CostMetric` (estimated cost derived from token usage, shared), and `LLMJudgeMetric` (1-5 relevance/faithfulness score via an LLM judge call, used by the QA app only).

#### Scenario: Exact match scoring
- **WHEN** `ExactMatchMetric.evaluate(case, output)` is called and the output's relevant field equals the case's expected value (within numeric tolerance where applicable)
- **THEN** the metric SHALL return a score of 1.0, otherwise 0.0

#### Scenario: Fuzzy match scoring
- **WHEN** `FuzzyMatchMetric.evaluate(case, output)` is called
- **THEN** the metric SHALL return a score between 0.0 and 1.0 based on normalized substring/token overlap between the output text and the expected answer

#### Scenario: LLM judge scoring is QA-only
- **WHEN** the extraction app's evaluation is aggregated
- **THEN** `LLMJudgeMetric` SHALL NOT be included; it SHALL only be included in the QA app's aggregation

### Requirement: Weighted score aggregation
The system SHALL aggregate a case's per-metric scores into a single weighted score, and aggregate all cases' weighted scores into a single run-level `aggregate_score`, using an application-specific weighting scheme (correctness-weighted higher than latency/cost).

#### Scenario: Aggregating a run's score
- **WHEN** all cases in a dataset have been scored by their applicable metrics
- **THEN** the system SHALL compute one `aggregate_score` for the run as the weighted mean of the metric scores across all cases, using weights that sum to 1.0

### Requirement: Retrieval metrics
The system SHALL provide two deterministic retrieval metrics for RAG-style apps: `retrieval_recall` (fraction of expected source snippets that were retrieved) and `citation_accuracy` (precision of cited snippets against expected sources).

#### Scenario: Retrieval recall scoring
- **WHEN** a case's expected output lists source indices and the app output lists retrieved indices
- **THEN** `retrieval_recall` SHALL equal the size of the intersection of the two sets divided by the number of expected source indices, and 0.0 when no expected source is retrieved

#### Scenario: Citation accuracy scoring
- **WHEN** an app output cites snippets
- **THEN** `citation_accuracy` SHALL equal the size of the intersection of cited and expected source indices divided by the number of cited indices
- **WHEN** the app output cites no snippets
- **THEN** `citation_accuracy` SHALL be 0.0

### Requirement: Weighted evaluation for six-metric apps
The system SHALL support evaluation of apps whose weight scheme combines six metrics, with the weights summing to 1.0.

#### Scenario: RAG weight scheme
- **WHEN** the evaluation engine scores a RAG run
- **THEN** the weights SHALL be fuzzy_match 0.20, llm_judge 0.20, retrieval_recall 0.25, citation_accuracy 0.10, latency 0.10, and cost 0.15, summing to 1.0

#### Scenario: Missing metric scores zero
- **WHEN** a case's output cannot be scored by a metric in the scheme
- **THEN** that metric SHALL contribute 0.0 to the case's weighted score, as with existing apps
