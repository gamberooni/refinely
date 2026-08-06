# Experiment Lineage Tracking Specification — Delta

## ADDED Requirements

### Requirement: Run history is listable
The system SHALL support listing an application's evaluation runs, newest first, with per-metric values joined onto each run row, so the CLI can render run history and comparisons without raw SQL.

#### Scenario: Listing runs for an application
- **WHEN** a caller invokes the lineage read API for an application name
- **THEN** the system SHALL return the app's runs ordered by `created_at` descending, each row containing run id, created-at timestamp, aggregate score, configuration, trial number (when present), and a mapping of metric name to value

#### Scenario: Limiting the run list
- **WHEN** a caller invokes the lineage read API for an application name with a limit
- **THEN** the system SHALL return at most that many runs

#### Scenario: Listing runs for an application with no history
- **WHEN** a caller invokes the lineage read API for an application name with no recorded runs
- **THEN** the system SHALL return an empty list
