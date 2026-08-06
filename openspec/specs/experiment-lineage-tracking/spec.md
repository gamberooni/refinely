# Experiment Lineage Tracking Specification

## Purpose

SQLite schema and read/write operations for `evaluation_runs`, `metric_results`, and `case_results`, giving reproducible lineage (app + dataset version + configuration + metrics) per run and per case. (Adapted from change `minimal-eval-optimization-prototype`.)
## Requirements
### Requirement: SQLite lineage schema
The system SHALL persist evaluation and optimization results to a SQLite database with three tables: `evaluation_runs` (run_id, app_name, dataset_version, configuration, optuna_trial_number, aggregate_score, created_at, tags), `metric_results` (run_id, metric_name, value), and `case_results` (run_id, case_id, input, output, expected, score, metric_scores, error). On databases created before `metric_scores`, `tags`, or `error` existed, initializing the schema SHALL backfill the columns without losing existing rows.

#### Scenario: Schema is created on first use
- **WHEN** the tracking module initializes against a database file that does not yet contain the lineage tables
- **THEN** it SHALL create all three tables (and allow Optuna's own internal tables to coexist in the same file) without erroring if the tables already exist

#### Scenario: Existing databases are upgraded
- **WHEN** the tracking module initializes against a database file whose `case_results` table predates the `metric_scores` column, or whose `evaluation_runs` predates `tags`, or whose `case_results` predates `error`
- **THEN** it SHALL add the missing columns while preserving existing rows, and existing rows SHALL read back with `tags` and `error` as NULL (absent) values

### Requirement: Run lineage is fully recorded
Every evaluation run (baseline or optimization trial) SHALL be recorded with enough information to reconstruct which application, dataset version, configuration, and tags produced which metrics — satisfying full experiment lineage (application + dataset version + configuration + metrics).

#### Scenario: Recording a completed run
- **WHEN** an `EvaluationRunner` run completes
- **THEN** the system SHALL insert one row into `evaluation_runs` (with a unique `run_id`, `app_name`, `dataset_version`, JSON-serialized `configuration`, `aggregate_score`, `created_at` timestamp, and the run's `tags` when provided), one row per metric into `metric_results` referencing that `run_id`, and one row per dataset case into `case_results` referencing that `run_id`

### Requirement: Lineage is queryable for comparison
The system SHALL support querying `evaluation_runs` to compare configurations by score, e.g. finding the best-scoring configuration for an application.

#### Scenario: Finding the best run for an application
- **WHEN** a query selects `evaluation_runs` filtered by `app_name` ordered by `aggregate_score` descending
- **THEN** the first row SHALL be the run with the highest `aggregate_score` for that application, and its `configuration` field SHALL be parseable back into the config dict used for that run

### Requirement: Case-level debugging visibility
The `case_results` table SHALL retain per-case input, output, expected value, score, per-metric score breakdown, and any per-case error so a user can inspect why a particular configuration scored as it did, without re-running the evaluation.

#### Scenario: Inspecting a low-scoring case
- **WHEN** a user queries `case_results` for a given `run_id` ordered by `score` ascending
- **THEN** the lowest-scoring rows SHALL contain the original `input`, the app's actual `output`, the `expected` value for that case, all stored as parseable JSON, and the per-metric `metric_scores` mapping of metric name to value for that case

#### Scenario: Reading a case that errored
- **WHEN** a case failed during a run (app failure or metric failure) and was captured in memory
- **THEN** the recorded `case_results` row SHALL carry the error message in its `error` column, and cases that ran cleanly SHALL carry a NULL `error` value

### Requirement: DSPy compile lineage
The system SHALL persist DSPy compile runs to a `dspy_compiles` table (compile_id, app_name, dataset_version, optimizer, config, artifact_path, baseline_score, compiled_score, created_at), created idempotently alongside the existing lineage tables, and SHALL support querying the best-scoring compile for an application.

#### Scenario: Recording a compile
- **WHEN** a compile pipeline run completes
- **THEN** the system SHALL insert one row into `dspy_compiles` with a unique `compile_id`, the app name, dataset version, optimizer identifier, JSON-serialized optimizer config, artifact path, baseline and compiled aggregate scores, and a created-at timestamp

#### Scenario: Finding the best compile for an application
- **WHEN** a query selects `dspy_compiles` filtered by `app_name` ordered by `compiled_score` descending
- **THEN** the first row SHALL be the compile with the highest `compiled_score` for that application, with its `artifact_path` pointing at the loadable compiled program

### Requirement: Run history is listable
The system SHALL support listing an application's evaluation runs, newest first, with per-metric values joined onto each run row, so the CLI can render run history and comparisons without raw SQL. Listing SHALL accept an optional tag filter that restricts results to runs bearing that tag.

#### Scenario: Listing runs for an application
- **WHEN** a caller invokes the lineage read API for an application name
- **THEN** the system SHALL return the app's runs ordered by `created_at` descending, each row containing run id, created-at timestamp, aggregate score, configuration, trial number (when present), tags (when present), and a mapping of metric name to value

#### Scenario: Limiting the run list
- **WHEN** a caller invokes the lineage read API for an application name with a limit
- **THEN** the system SHALL return at most that many runs

#### Scenario: Listing runs filtered by tag
- **WHEN** a caller invokes the lineage read API for an application name with a tag filter
- **THEN** the system SHALL return only the runs bearing that tag, newest first, and SHALL return an empty list when no run bears the tag

#### Scenario: Listing runs for an application with no history
- **WHEN** a caller invokes the lineage read API for an application name with no recorded runs
- **THEN** the system SHALL return an empty list

