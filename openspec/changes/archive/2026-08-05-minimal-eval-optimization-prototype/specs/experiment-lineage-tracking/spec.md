## ADDED Requirements

### Requirement: SQLite lineage schema
The system SHALL persist evaluation and optimization results to a SQLite database with three tables: `evaluation_runs` (run_id, app_name, dataset_version, configuration, optuna_trial_number, aggregate_score, created_at), `metric_results` (run_id, metric_name, value), and `case_results` (run_id, case_id, input, output, expected, score).

#### Scenario: Schema is created on first use
- **WHEN** the tracking module initializes against a database file that does not yet contain the lineage tables
- **THEN** it SHALL create all three tables (and allow Optuna's own internal tables to coexist in the same file) without erroring if the tables already exist

### Requirement: Run lineage is fully recorded
Every evaluation run (baseline or optimization trial) SHALL be recorded with enough information to reconstruct which application, dataset version, and configuration produced which metrics — satisfying full experiment lineage (application + dataset version + configuration + metrics).

#### Scenario: Recording a completed run
- **WHEN** an `EvaluationRunner` run completes
- **THEN** the system SHALL insert one row into `evaluation_runs` (with a unique `run_id`, `app_name`, `dataset_version`, JSON-serialized `configuration`, `aggregate_score`, and `created_at` timestamp), one row per metric into `metric_results` referencing that `run_id`, and one row per dataset case into `case_results` referencing that `run_id`

### Requirement: Lineage is queryable for comparison
The system SHALL support querying `evaluation_runs` to compare configurations by score, e.g. finding the best-scoring configuration for an application.

#### Scenario: Finding the best run for an application
- **WHEN** a query selects `evaluation_runs` filtered by `app_name` ordered by `aggregate_score` descending
- **THEN** the first row SHALL be the run with the highest `aggregate_score` for that application, and its `configuration` field SHALL be parseable back into the config dict used for that run

### Requirement: Case-level debugging visibility
The `case_results` table SHALL retain per-case input, output, expected value, and score so a user can inspect why a particular configuration scored as it did, without re-running the evaluation.

#### Scenario: Inspecting a low-scoring case
- **WHEN** a user queries `case_results` for a given `run_id` ordered by `score` ascending
- **THEN** the lowest-scoring rows SHALL contain the original `input`, the app's actual `output`, and the `expected` value for that case, all stored as parseable JSON
