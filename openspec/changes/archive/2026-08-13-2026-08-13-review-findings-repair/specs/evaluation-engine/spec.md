# Evaluation Engine Specification — Delta

## MODIFIED Requirements

### Requirement: Metric plugins

The system SHALL provide the following metric implementations, each computing a numeric score for a single case: `ExactMatchMetric` (field-level equality, used by the extraction app), `FuzzyMatchMetric` (normalized substring/token-overlap match, used by QA and RAG), `LatencyMetric` (wall-clock duration per case, shared), `CostMetric` (estimated cost derived from token usage, shared), and `LLMJudgeMetric` (a context-grounded 0.0–1.0 score via an LLM judge call, used by the QA and RAG apps).

#### Scenario: LLM judge scoring is context-grounded
- **WHEN** `LLMJudgeMetric.evaluate(case, output)` is called for a QA/RAG case
- **THEN** the judge prompt SHALL contain the question, the answer, and the retrieved context only — the expected answer SHALL NOT appear in the prompt — and the metric SHALL return a 0.0–1.0 score with a one-line rationale, parsed robustly (fence stripping + repair retry on parse failure)

#### Scenario: LLM judge is not used for extraction
- **WHEN** the extraction app's evaluation is aggregated
- **THEN** `LLMJudgeMetric` SHALL NOT be included; the extraction composite SHALL be `exact_match 0.7, latency 0.15, cost 0.15`

### Requirement: Weighted score aggregation

The system SHALL aggregate a case's per-metric scores into a single weighted score, and aggregate all cases' weighted scores into a single run-level `aggregate_score`, using an application-specific weighting scheme (correctness-weighted higher than latency/cost). When a metric throws for a case, that metric SHALL contribute 0.0 for that case in BOTH the per-metric run-level means and the weighted aggregate, and the run result SHALL report the count of errored cases.

#### Scenario: Aggregating a run's score
- **WHEN** all cases in a dataset have been scored by their applicable metrics
- **THEN** the system SHALL compute one `aggregate_score` for the run as the weighted mean of the metric scores across all cases, using weights that sum to 1.0

#### Scenario: A throwing metric is consistent with the aggregate
- **WHEN** a metric throws for some cases in a run
- **THEN** the run-level metric mean for that metric SHALL count the failed cases as 0.0 (identical to the aggregate's treatment), the run SHALL report how many cases errored, and the mean SHALL equal the value implied by the aggregate

## ADDED Requirements

### Requirement: Judge model decoupling and identity freeze

The system SHALL evaluate the LLM judge with a judge model that is independent of the generator model: a `--judge-model` option on evaluate/optimize/compile backed by a `judge_model` setting, with the default rule that the judge model SHALL NOT equal the generator model unless explicitly overridden (an explicit override SHALL produce a loud warning). The judge's identity SHALL be frozen per run: `evaluation_runs` SHALL record the judge model and the judge prompt version.

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
