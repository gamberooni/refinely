# Lineage CLI Read-Back Specification

Delta for `actionable-results`: `show --run` renders a per-case error column and an errored count; `show`/`compare`/`export` accept a `--tag` filter; `compare` gains `--diff-config` (config delta section) and `--cases` (paired per-case regression table).

## MODIFIED Requirements

### Requirement: Show run history
The system SHALL provide a `show` subcommand (`refinely show <app>`) that renders a table of the app's recorded evaluation runs, newest first, including run id, creation time, aggregate score, per-metric values, and the Optuna trial number when the run was a trial. The output SHALL also summarize the best-scoring run (config included) and the best-scoring DSPy compile for the app, when present. An optional `--tag <tag>` filter SHALL restrict the rendered runs to those bearing that tag.

#### Scenario: Showing runs for an app with history
- **WHEN** a user runs `refinely show <app>` and the lineage database contains recorded runs for that app
- **THEN** the CLI SHALL print a table of runs ordered by `created_at` descending (newest first), one row per run with run id, created-at timestamp, aggregate score, each metric value, and trial number (blank when not a trial)

#### Scenario: Showing runs for an app with no history
- **WHEN** a user runs `refinely show <app>` and the lineage database contains no runs for that app
- **THEN** the CLI SHALL print a message stating no runs were found for the app

#### Scenario: Showing best run and best compile summaries
- **WHEN** a user runs `refinely show <app>` and the app has at least one recorded run or compile
- **THEN** the output SHALL include the best run's run id, aggregate score, and configuration, and (when a compile exists) the best compile's compile id and compiled score

#### Scenario: Filtering run history by tag
- **WHEN** a user runs `refinely show <app> --tag <tag>`
- **THEN** the CLI SHALL render only the runs bearing that tag
- **WHEN** no run bears the tag
- **THEN** the CLI SHALL print a message stating no runs were found matching the tag

### Requirement: Show per-case results for a run
The system SHALL support `refinely show <app> --run <run_id>` to render the run's per-case results as a table, ordered by case score ascending (worst cases first), with case id, score, input, expected, output, and any per-case error. The output SHALL report how many cases errored.

#### Scenario: Drilling into a run's cases
- **WHEN** a user runs `refinely show <app> --run <run_id>` and the run has recorded case results
- **THEN** the CLI SHALL print a table of the run's cases ordered by score ascending, with case id, score, input, expected, and output

#### Scenario: Rendering errored cases
- **WHEN** a user runs `refinely show <app> --run <run_id>` and some of the run's cases have a recorded error
- **THEN** the CLI SHALL render an error column populated with each errored case's message, print a summary stating the number of errored cases, and SHALL render clean cases with a blank error value

#### Scenario: Unknown run id
- **WHEN** a user runs `refinely show <app> --run <run_id>` and no run with that id exists
- **THEN** the CLI SHALL exit with a clear error stating the run id was not found

### Requirement: Compare runs against a baseline
The system SHALL provide a `compare` subcommand (`refinely compare <app>`) that renders a table comparing each of the app's runs against a baseline run, showing per-metric deltas. The default baseline SHALL be the run immediately preceding each row in chronological order (previous-run comparison); the first run SHALL be marked as the baseline with no deltas. An optional `--baseline <run_id>` flag SHALL override the baseline to a specific run, in which case every row SHALL show deltas against that run. The command SHALL accept a `--tag <tag>` filter restricting the compared runs, and optional `--diff-config` and `--cases` flags that extend the output with a configuration delta section and a per-case paired comparison respectively.

#### Scenario: Comparing runs against the previous run
- **WHEN** a user runs `refinely compare <app>` with no baseline flag
- **THEN** the CLI SHALL print a table ordered by `created_at` ascending where each run row shows its metric values and, for every run except the first, deltas against the immediately preceding run

#### Scenario: Comparing runs against an explicit baseline
- **WHEN** a user runs `refinely compare <app> --baseline <run_id>`
- **THEN** the CLI SHALL print the same table where every run row shows deltas against the specified baseline run

#### Scenario: Unknown baseline run id
- **WHEN** a user runs `refinely compare <app> --baseline <run_id>` and no run with that id exists
- **THEN** the CLI SHALL exit with a clear error stating the baseline run id was not found

#### Scenario: Comparing only tagged runs
- **WHEN** a user runs `refinely compare <app> --tag <tag>`
- **THEN** the CLI SHALL compare only the runs bearing that tag
- **WHEN** fewer than two runs bear the tag
- **THEN** the CLI SHALL print a message stating that comparison needs at least two matching runs

### Requirement: Export runs to a file
The system SHALL provide an `export` subcommand (`refinely export <app>`) that writes the app's runs and their metric values to a file. The `--format` option SHALL accept `csv` or `json` (default `csv`); the `--output` option SHALL set the output path (default `<app>_runs.csv` or `<app>_runs.json` matching the format, in the current directory). An optional `--tag <tag>` filter SHALL restrict the exported runs to those bearing that tag. The command SHALL always write a file and report its path.

#### Scenario: Exporting runs as CSV
- **WHEN** a user runs `refinely export <app>` without a `--format` flag and the app has recorded runs
- **THEN** the CLI SHALL write a CSV file at `<app>_runs.csv` (or the path given by `--output`) containing one row per run with run id, created-at timestamp, aggregate score, trial number, and one column per metric, and SHALL print the output path

#### Scenario: Exporting runs as JSON
- **WHEN** a user runs `refinely export <app> --format json --output <path>`
- **THEN** the CLI SHALL write a JSON file at the given path containing a list of run objects (each with run id, created-at timestamp, aggregate score, trial number, and metric values) and SHALL print the output path

#### Scenario: Exporting only tagged runs
- **WHEN** a user runs `refinely export <app> --tag <tag>`
- **THEN** the CLI SHALL write the file containing only the runs bearing that tag

#### Scenario: Exporting an app with no runs
- **WHEN** a user runs `refinely export <app>` and the lineage database contains no runs for that app
- **THEN** the CLI SHALL write the file with an empty row set and SHALL print the output path

## ADDED Requirements

### Requirement: Compare configurations between runs
When `compare` is invoked with `--diff-config`, the output SHALL include a configuration delta section alongside the per-metric deltas, showing which configuration keys changed between the baseline run and each compared run.

#### Scenario: Rendering the config delta section
- **WHEN** a user runs `refinely compare <app> --diff-config` and the compared runs' configurations differ
- **THEN** the CLI SHALL render, for each compared run, the configuration keys that were added, removed, or changed in value relative to the baseline

#### Scenario: Identical configurations produce no delta
- **WHEN** a user runs `refinely compare <app> --diff-config` and a compared run's configuration equals the baseline's
- **THEN** the CLI SHALL render no config delta for that run (or an explicit no-change marker)

### Requirement: Compare per-case results between runs
When `compare` is invoked with `--cases`, the output SHALL include a paired per-case comparison between the two compared runs (the baseline and the most recent run), rendering a table of per-case metric deltas with a direction of broke, fixed, or unchanged, and a summary line of the form "N broke / M fixed / K unchanged". When the two runs' `dataset_version` values differ, the CLI SHALL warn that index-based case pairing may be meaningless.

#### Scenario: Rendering the per-case comparison
- **WHEN** a user runs `refinely compare <app> --cases` and both the baseline and the most recent run have recorded case results
- **THEN** the CLI SHALL render a table pairing cases by index, showing for each the case id, metric score before, metric score after, and delta, and SHALL print a summary of the form "N broke / M fixed / K unchanged"

#### Scenario: Cases across different dataset versions
- **WHEN** a user runs `refinely compare <app> --cases` and the two runs' `dataset_version` values differ
- **THEN** the CLI SHALL print a warning that case pairing by index may be meaningless across dataset versions

#### Scenario: A run with no case results
- **WHEN** a user runs `refinely compare <app> --cases` and one of the two runs has no recorded case results
- **THEN** the CLI SHALL print a message stating per-case comparison is not possible for those runs
