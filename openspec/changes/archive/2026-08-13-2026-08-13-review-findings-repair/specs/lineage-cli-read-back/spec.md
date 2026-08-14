# Lineage CLI Read-back Specification — Delta

## MODIFIED Requirements

### Requirement: Compare per-case results between runs

When `compare` is invoked with `--cases`, the output SHALL include a paired per-case comparison between the two compared runs (the baseline and the most recent run), rendering a table of per-case metric deltas with a direction of broke, fixed, or unchanged, and a summary line of the form "N broke / M fixed / K unchanged". Case pairing SHALL be by case identity (case id), not by score rank. When the two runs' `dataset_version` values differ, the CLI SHALL warn that cross-version pairing may be meaningless.

#### Scenario: Rendering the per-case comparison
- **WHEN** a user runs `refinely compare <app> --cases` and both the baseline and the most recent run have recorded case results
- **THEN** the CLI SHALL render a table pairing cases by case identity (id), showing for each the case id, metric score before, metric score after, and delta, and SHALL print a summary of the form "N broke / M fixed / K unchanged"

#### Scenario: Equal-score cases do not cross-pair
- **WHEN** two runs contain cases with identical metric scores
- **THEN** the CLI SHALL pair cases by id regardless of score order, so equal scores never swap pairings

#### Scenario: Cases present in only one run
- **WHEN** a case id exists in one run's case results but not the other's
- **THEN** the CLI SHALL report it as unmatched rather than pairing it with a different case

#### Scenario: Cases across different dataset versions
- **WHEN** a user runs `refinely compare <app> --cases` and the two runs' `dataset_version` values differ
- **THEN** the CLI SHALL print a warning that case pairing by id may be meaningless across dataset versions

#### Scenario: A run with no case results
- **WHEN** a user runs `refinely compare <app> --cases` and one of the two runs has no recorded case results
- **THEN** the CLI SHALL print a message stating per-case comparison is not possible for those runs

### Requirement: Export runs to a file

The system SHALL provide an `export` subcommand (`refinely export <app>`) that writes the app's runs and their metric values to a file. The `--format` option SHALL accept `csv` or `json` (default `csv`); the `--output` option SHALL set the output path (default `<app>_runs.csv` or `<app>_runs.json` matching the format, in the current directory). An optional `--tag <tag>` filter SHALL restrict the exported runs to those bearing that tag, applied before any row limit. The export SHALL contain ALL matching runs — the system SHALL NOT truncate to a fixed row count. The command SHALL always write a file and report its path.

#### Scenario: Exporting runs as CSV
- **WHEN** a user runs `refinely export <app>` without a `--format` flag and the app has recorded runs
- **THEN** the CLI SHALL write a CSV file at `<app>_runs.csv` (or the path given by `--output`) containing one row per run with run id, created-at timestamp, aggregate score, trial number, and one column per metric, and SHALL print the output path

#### Scenario: Exporting runs as JSON
- **WHEN** a user runs `refinely export <app> --format json --output <path>`
- **THEN** the CLI SHALL write a JSON file at the given path containing a list of run objects (each with run id, created-at timestamp, aggregate score, trial number, and metric values) and SHALL print the output path

#### Scenario: Exporting only tagged runs
- **WHEN** a user runs `refinely export <app> --tag <tag>`
- **THEN** the CLI SHALL write the file containing only the runs bearing that tag

#### Scenario: Exporting more than fifty runs
- **WHEN** an app has more than fifty runs (tagged or not) matching the export filter
- **THEN** the CLI SHALL write all of them; no silent truncation to 50 rows SHALL occur

#### Scenario: Exporting an app with no runs
- **WHEN** a user runs `refinely export <app>` and the lineage database contains no runs for that app
- **THEN** the CLI SHALL write the file with an empty row set and SHALL print the output path
