# CLI Specification — Delta

## ADDED Requirements

### Requirement: Show run history via CLI
The CLI SHALL support a `show` subcommand (`refinely show <app>`) that reads the lineage database and renders the app's run history, and (`refinely show <app> --run <run_id>`) per-case results for a specific run.

#### Scenario: Running show for an app
- **WHEN** a user runs `refinely show <app>`
- **THEN** the CLI SHALL render a table of the app's recorded runs, newest first, with best run and best compile summaries

#### Scenario: Running show with a run id
- **WHEN** a user runs `refinely show <app> --run <run_id>`
- **THEN** the CLI SHALL render the run's per-case results, worst cases first

### Requirement: Compare runs via CLI
The CLI SHALL support a `compare` subcommand (`refinely compare <app> [--baseline <run_id>]`) that renders the app's runs with per-metric deltas against a baseline run.

#### Scenario: Running compare for an app
- **WHEN** a user runs `refinely compare <app>`
- **THEN** the CLI SHALL render the app's runs in chronological order with per-metric deltas against the previous run

### Requirement: Export runs via CLI
The CLI SHALL support an `export` subcommand (`refinely export <app> [--format csv|json] [--output FILE]`) that writes the app's runs and metric values to a file.

#### Scenario: Running export for an app
- **WHEN** a user runs `refinely export <app>`
- **THEN** the CLI SHALL write the app's runs to a CSV file and print its path

### Requirement: Rich-formatted terminal output
The CLI SHALL render command output using `rich` formatting — panels and tables — for the `evaluate`, `optimize`, `compile`, `show`, `compare`, and `export` commands, keeping the printed data content unchanged for the existing commands.

#### Scenario: Existing commands print rich-formatted output
- **WHEN** a user runs any of `evaluate`, `optimize`, `compile`, `show`, `compare`, or `export`
- **THEN** the output SHALL be rendered with rich panels/tables and SHALL contain the same data values the command previously printed as plain text
