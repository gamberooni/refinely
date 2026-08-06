# Experiment Lineage Tracking Specification — Delta

## MODIFIED Requirements

### Requirement: SQLite lineage schema

The system SHALL persist evaluation and optimization results to a SQLite database with three tables: `evaluation_runs` (run_id, app_name, dataset_version, configuration, model_name, optuna_trial_number, aggregate_score, created_at), `metric_results` (run_id, metric_name, value), and `case_results` (run_id, case_id, input, output, expected, score, metric_scores). On databases created before `metric_scores` or `model_name` existed, initializing the schema SHALL add the missing columns without losing existing rows.

#### Scenario: Schema is created on first use
- **WHEN** the tracking module initializes against a database file that does not yet contain the lineage tables
- **THEN** it SHALL create all three tables (and allow Optuna's own internal tables to coexist in the same file) without erroring if the tables already exist

#### Scenario: Existing databases are upgraded
- **WHEN** the tracking module initializes against a database file whose `case_results` table predates the `metric_scores` column
- **THEN** it SHALL add the missing column while preserving existing rows

#### Scenario: Databases predating the model column are upgraded
- **WHEN** the tracking module initializes against a database file whose `evaluation_runs` table predates the `model_name` column
- **THEN** it SHALL add the `model_name` column (nullable) while preserving existing rows

### Requirement: Run lineage is fully recorded

Every evaluation run (baseline or optimization trial) SHALL be recorded with enough information to reconstruct which application, dataset version, model, and configuration produced which metrics — satisfying full experiment lineage (application + dataset version + configuration + model + metrics).

#### Scenario: Recording a completed run
- **WHEN** an `EvaluationRunner` run completes
- **THEN** the system SHALL insert one row into `evaluation_runs` (with a unique `run_id`, `app_name`, `dataset_version`, JSON-serialized `configuration`, `model_name`, `aggregate_score`, and `created_at` timestamp), one row per metric into `metric_results` referencing that `run_id`, and one row per dataset case into `case_results` referencing that `run_id`

### Requirement: Run history is listable

The system SHALL support listing an application's evaluation runs, newest first, with per-metric values joined onto each run row, so the CLI can render run history and comparisons without raw SQL.

#### Scenario: Listing runs for an application
- **WHEN** a caller invokes the lineage read API for an application name
- **THEN** the system SHALL return the app's runs ordered by `created_at` descending, each row containing run id, created-at timestamp, aggregate score, configuration, model name, trial number (when present), and a mapping of metric name to value

#### Scenario: Limiting the run list
- **WHEN** a caller invokes the lineage read API for an application name with a limit
- **THEN** the system SHALL return at most that many runs

#### Scenario: Listing runs for an application with no history
- **WHEN** a caller invokes the lineage read API for an application name with no recorded runs
- **THEN** the system SHALL return an empty list
