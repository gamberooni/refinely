# Lineage CLI Read-Back Specification

## Purpose

CLI commands that surface lineage database contents to the terminal and to files: run history, per-case drill-down, run comparison, and export — so experiment data is accessible without raw `sqlite3` queries.
## Requirements
### Requirement: Show run history
The system SHALL provide a `show` subcommand (`crucible show <app>`) that renders a table of the app's recorded evaluation runs, newest first, including run id, creation time, aggregate score, per-metric values, and the Optuna trial number when the run was a trial. The output SHALL also summarize the best-scoring run (config included) and the best-scoring DSPy compile for the app, when present.

#### Scenario: Showing runs for an app with history
- **WHEN** a user runs `crucible show <app>` and the lineage database contains recorded runs for that app
- **THEN** the CLI SHALL print a table of runs ordered by `created_at` descending (newest first), one row per run with run id, created-at timestamp, aggregate score, each metric value, and trial number (blank when not a trial)

#### Scenario: Showing runs for an app with no history
- **WHEN** a user runs `crucible show <app>` and the lineage database contains no runs for that app
- **THEN** the CLI SHALL print a message stating no runs were found for the app

#### Scenario: Showing best run and best compile summaries
- **WHEN** a user runs `crucible show <app>` and the app has at least one recorded run or compile
- **THEN** the output SHALL include the best run's run id, aggregate score, and configuration, and (when a compile exists) the best compile's compile id and compiled score

### Requirement: Show per-case results for a run
The system SHALL support `crucible show <app> --run <run_id>` to render the run's per-case results as a table, ordered by case score ascending (worst cases first), with case id, score, input, expected, and output.

#### Scenario: Drilling into a run's cases
- **WHEN** a user runs `crucible show <app> --run <run_id>` and the run has recorded case results
- **THEN** the CLI SHALL print a table of the run's cases ordered by score ascending, with case id, score, input, expected, and output

#### Scenario: Unknown run id
- **WHEN** a user runs `crucible show <app> --run <run_id>` and no run with that id exists
- **THEN** the CLI SHALL exit with a clear error stating the run id was not found

### Requirement: Compare runs against a baseline
The system SHALL provide a `compare` subcommand (`crucible compare <app>`) that renders a table comparing each of the app's runs against a baseline run, showing per-metric deltas. The default baseline SHALL be the run immediately preceding each row in chronological order (previous-run comparison); the first run SHALL be marked as the baseline with no deltas. An optional `--baseline <run_id>` flag SHALL override the baseline to a specific run, in which case every row SHALL show deltas against that run.

#### Scenario: Comparing runs against the previous run
- **WHEN** a user runs `crucible compare <app>` with no baseline flag
- **THEN** the CLI SHALL print a table ordered by `created_at` ascending where each run row shows its metric values and, for every run except the first, deltas against the immediately preceding run

#### Scenario: Comparing runs against an explicit baseline
- **WHEN** a user runs `crucible compare <app> --baseline <run_id>`
- **THEN** the CLI SHALL print the same table where every run row shows deltas against the specified baseline run

#### Scenario: Unknown baseline run id
- **WHEN** a user runs `crucible compare <app> --baseline <run_id>` and no run with that id exists
- **THEN** the CLI SHALL exit with a clear error stating the baseline run id was not found

### Requirement: Export runs to a file
The system SHALL provide an `export` subcommand (`crucible export <app>`) that writes the app's runs and their metric values to a file. The `--format` option SHALL accept `csv` or `json` (default `csv`); the `--output` option SHALL set the output path (default `<app>_runs.csv` or `<app>_runs.json` matching the format, in the current directory). The command SHALL always write a file and report its path.

#### Scenario: Exporting runs as CSV
- **WHEN** a user runs `crucible export <app>` without a `--format` flag and the app has recorded runs
- **THEN** the CLI SHALL write a CSV file at `<app>_runs.csv` (or the path given by `--output`) containing one row per run with run id, created-at timestamp, aggregate score, trial number, and one column per metric, and SHALL print the output path

#### Scenario: Exporting runs as JSON
- **WHEN** a user runs `crucible export <app> --format json --output <path>`
- **THEN** the CLI SHALL write a JSON file at the given path containing a list of run objects (each with run id, created-at timestamp, aggregate score, trial number, and metric values) and SHALL print the output path

#### Scenario: Exporting an app with no runs
- **WHEN** a user runs `crucible export <app>` and the lineage database contains no runs for that app
- **THEN** the CLI SHALL write the file with an empty row set and SHALL print the output path

### Requirement: Invalid export format
The `export` subcommand's `--format` option SHALL reject values other than `csv` and `json` with a clear error.

#### Scenario: Unsupported format value
- **WHEN** a user runs `crucible export <app> --format yaml`
- **THEN** the CLI SHALL exit with a clear error listing the supported formats
