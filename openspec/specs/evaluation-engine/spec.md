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

The system SHALL provide the following metric implementations, each computing a numeric score for a single case: `ExactMatchMetric` (field-level equality, used by the extraction app), `FuzzyMatchMetric` (normalized substring/token-overlap match, used by QA and RAG), `LatencyMetric` (wall-clock duration per case, shared), `CostMetric` (estimated cost derived from token usage, shared), and `LLMJudgeMetric` (a context-grounded 0.0–1.0 score via an LLM judge call, used by the QA and RAG apps).

#### Scenario: LLM judge scoring is context-grounded
- **WHEN** `LLMJudgeMetric.evaluate(case, output)` is called for a QA/RAG case
- **THEN** the judge prompt SHALL contain the question, the answer, and the retrieved context only — the expected answer SHALL NOT appear in the prompt — and the metric SHALL return a 0.0–1.0 score with a one-line rationale, parsed robustly (fence stripping + repair retry on parse failure)

#### Scenario: LLM judge is not used for extraction
- **WHEN** the extraction app's evaluation is aggregated
- **THEN** `LLMJudgeMetric` SHALL NOT be included; the extraction composite SHALL be `exact_match 0.7, latency 0.15, cost 0.15`

### Requirement: Weighted score aggregation

The system SHALL aggregate a case's per-metric scores into a single weighted score, and aggregate all cases' weighted scores into a single run-level `aggregate_score`, using an application-specific weighting scheme (correctness-weighted higher than latency/cost). When a metric throws for a case, that metric SHALL contribute 0.0 for that case in BOTH the per-metric run-level means and the weighted aggregate, and the run result SHALL report the count of errored cases. When a metric raises `MetricUnavailableError` (its measurement is not possible for the case, e.g. cost on a run with no token usage), that metric SHALL be excluded from the case's weighted score and from the run-level metric mean, and the aggregate SHALL renormalize the remaining weights to sum to 1.0.

#### Scenario: Aggregating a run's score
- **WHEN** all cases in a dataset have been scored by their applicable metrics
- **THEN** the system SHALL compute one `aggregate_score` for the run as the weighted mean of the metric scores across all cases, using weights that sum to 1.0

#### Scenario: A throwing metric is consistent with the aggregate
- **WHEN** a metric throws for some cases in a run
- **THEN** the run-level metric mean for that metric SHALL count the failed cases as 0.0 (identical to the aggregate's treatment), the run SHALL report how many cases errored, and the mean SHALL equal the value implied by the aggregate

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

#### Scenario: A metric that throws scores zero
- **WHEN** a metric throws while scoring a case
- **THEN** that metric SHALL contribute 0.0 to the case's weighted score and to the run-level mean, as with existing apps

#### Scenario: An unavailable metric is excluded and renormalized
- **WHEN** a metric raises `MetricUnavailableError` for a case (e.g. cost on a compiled run with no token usage)
- **THEN** that metric SHALL be excluded from the case's weighted score, the run-level metric mean SHALL be marked `n/a` in lineage and read-back, and the aggregate SHALL renormalize the remaining weights to sum to 1.0

### Requirement: Judge model decoupling and identity freeze

The system SHALL evaluate the LLM judge with a judge model that is independent of the generator model: a `--judge-model` option on evaluate/optimize/compile backed by a `judge_model` setting. When the effective judge model equals the generator model and no explicit override was passed, the CLI SHALL print a loud warning; an explicit `--judge-model` override SHALL suppress the warning. The judge's identity SHALL be frozen per run: `evaluation_runs` SHALL record the judge model and the judge prompt version.

#### Scenario: Judge model defaults to a different model
- **WHEN** an evaluation runs without `--judge-model`
- **THEN** the judge SHALL use `settings.judge_model`, and if that equals the app's generator model the CLI SHALL warn loudly unless an explicit override was passed

#### Scenario: Judge identity is recorded
- **WHEN** an evaluation run completes
- **THEN** the run's lineage record SHALL contain the judge model name and the judge prompt version used

### Requirement: Judge consistency reporting

The system SHALL provide a judge-consistency check that re-scores a sample of cases with two judge calls at a nonzero temperature and reports the agreement rate, so judge noise is visible.

#### Scenario: Reporting judge self-consistency
- **WHEN** the consistency check runs over a sample of cases
- **THEN** the check SHALL report the fraction of cases where the two judge calls agree within tolerance, alongside the sample size

