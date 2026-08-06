# CLI Specification

Delta for `dev-ergonomics`: the CLI gains three developer-focused subcommands — `new app`, `doctor`, and `dataset stats` — alongside the existing evaluate/optimize/compile/show/compare/export surface.

## ADDED Requirements

### Requirement: Scaffold a new app via CLI
The CLI SHALL support a `new app` subcommand (`crucible new app <name> [--dataset <path>]`) that scaffolds an app module and dataset stub without modifying `pyproject.toml`, printing the entry-point declaration line the user must add.

#### Scenario: Scaffolding a new app
- **WHEN** a user runs `crucible new app invoices`
- **THEN** the CLI SHALL write `apps/invoices.py` and `datasets/invoices_v1.json`, and print the `[project.entry-points."crucible.apps"]` line to add to `pyproject.toml`

### Requirement: Run doctor via CLI
The CLI SHALL support a `doctor` subcommand (`crucible doctor [--network]`) that runs deterministic health checks (app discovery, dataset loading, DB schema, API key) and reports pass/fail per check with fix hints, exiting non-zero when any check fails.

#### Scenario: Running doctor on a healthy setup
- **WHEN** a user runs `crucible doctor` and all checks pass
- **THEN** the CLI SHALL print a per-check pass report and exit with status 0

#### Scenario: Running doctor on a broken setup
- **WHEN** a user runs `crucible doctor` and a check fails
- **THEN** the CLI SHALL print the failing check with a fix hint and exit with a non-zero status

### Requirement: Show dataset stats via CLI
The CLI SHALL support a `dataset stats` subcommand (`crucible dataset stats <app>`) that renders case count, file size, input/expected shape summaries, and a malformed-case report for the app's dataset.

#### Scenario: Running dataset stats
- **WHEN** a user runs `crucible dataset stats <app>`
- **THEN** the CLI SHALL print the dataset's case count, file size, shape summaries, and malformed-case report (or a clear parse error naming the file and failing case)
