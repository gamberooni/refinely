# Developer Tools Specification — Delta

## ADDED Requirements

### Requirement: Scaffold a new app
The system SHALL provide `refinely new app <name> [--dataset <path>]` that creates an `apps/<name>.py` module containing a working `register_app` call with placeholders for `build_adapter`, `metrics_factory`, `search_space`, `default_config`, `weights`, and `dataset_path`, and a `datasets/<name>_v1.json` stub (empty case list). The command SHALL NOT modify `pyproject.toml`; it SHALL print the entry-point declaration line the user must add.

#### Scenario: Scaffolding an app module
- **WHEN** a user runs `refinely new app invoices`
- **THEN** the CLI SHALL write `apps/invoices.py` with a `register_app(...)` skeleton covering all six required fields, write `datasets/invoices_v1.json` containing a valid empty dataset, and print the `[project.entry-points."refinely.apps"]` line to add to `pyproject.toml`

#### Scenario: Scaffolding with an existing dataset path
- **WHEN** a user runs `refinely new app invoices --dataset /path/to/data.json`
- **THEN** the CLI SHALL write the app module with `dataset_path` pointing at the given path and SHALL NOT create a dataset stub

#### Scenario: Invalid app name
- **WHEN** a user runs `refinely new app <name>` with a name that is not a valid Python module identifier
- **THEN** the CLI SHALL exit with a clear error and write no files

#### Scenario: Scaffolding into an existing file
- **WHEN** a user runs `refinely new app invoices` and `apps/invoices.py` already exists
- **THEN** the CLI SHALL exit with a clear error and not overwrite the existing file

### Requirement: Doctor health checks
The system SHALL provide `refinely doctor` that runs deterministic checks without network access: app discovery succeeds via `discover_apps()`, each registered app's dataset loads, the lineage database schema initializes cleanly, and an API key is present in settings. An opt-in `--network` flag SHALL additionally probe the configured API endpoint. When any check fails, the CLI SHALL print the failing check with a fix hint and exit with a non-zero status; when all pass, it SHALL exit zero.

#### Scenario: Healthy installation
- **WHEN** a user runs `refinely doctor` and apps discover, datasets load, the DB schema initializes, and an API key is configured
- **THEN** the CLI SHALL print a per-check pass report and exit with status 0

#### Scenario: Failing check with hint
- **WHEN** a user runs `refinely doctor` and one or more checks fail (e.g. a dataset file is missing, the DB cannot initialize, or no API key is set)
- **THEN** the CLI SHALL print each failing check with a concrete fix hint and exit with a non-zero status

#### Scenario: Optional network probe
- **WHEN** a user runs `refinely doctor --network`
- **THEN** the CLI SHALL additionally attempt a connectivity probe to the configured base URL and report its result as a check

#### Scenario: No network by default
- **WHEN** a user runs `refinely doctor` without `--network`
- **THEN** the CLI SHALL make no network calls

### Requirement: Dataset statistics
The system SHALL provide `refinely dataset stats <app>` that renders, for the app's dataset: case count, file size in bytes, a summary of the shape of `input` fields and `expected` values across cases, and a malformed-case report identifying cases with structural inconsistencies. Malformed cases SHALL be identified and listed without raising.

#### Scenario: Reporting dataset statistics
- **WHEN** a user runs `refinely dataset stats <app>` and the app's dataset loads
- **THEN** the CLI SHALL print the case count, file size, per-field input shape summary (e.g. which keys are present and how often), and expected-value shape summary

#### Scenario: Reporting malformed cases
- **WHEN** a user runs `refinely dataset stats <app>` and some cases are structurally inconsistent (e.g. a case missing keys present in the majority of cases, or an `expected` of a different type than the norm)
- **THEN** the CLI SHALL list the affected case ids in a malformed-case report

#### Scenario: Dataset that fails to parse
- **WHEN** a user runs `refinely dataset stats <app>` and the dataset file is missing, not valid JSON, or contains cases failing `load_dataset` validation
- **THEN** the CLI SHALL exit with a clear error naming the file and the failing case, without printing statistics
