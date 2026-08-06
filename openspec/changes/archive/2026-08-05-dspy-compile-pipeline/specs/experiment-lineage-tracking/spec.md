## ADDED Requirements

### Requirement: DSPy compile lineage
The system SHALL persist DSPy compile runs to a `dspy_compiles` table (compile_id, app_name, dataset_version, optimizer, config, artifact_path, baseline_score, compiled_score, created_at), created idempotently alongside the existing lineage tables, and SHALL support querying the best-scoring compile for an application.

#### Scenario: Recording a compile
- **WHEN** a compile pipeline run completes
- **THEN** the system SHALL insert one row into `dspy_compiles` with a unique `compile_id`, the app name, dataset version, optimizer identifier, JSON-serialized optimizer config, artifact path, baseline and compiled aggregate scores, and a created-at timestamp

#### Scenario: Finding the best compile for an application
- **WHEN** a query selects `dspy_compiles` filtered by `app_name` ordered by `compiled_score` descending
- **THEN** the first row SHALL be the compile with the highest `compiled_score` for that application, with its `artifact_path` pointing at the loadable compiled program
