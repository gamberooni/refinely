# CLI Specification — Delta

## MODIFIED Requirements

### Requirement: Run evaluation via CLI

The CLI SHALL support running a baseline evaluation for a single named application against its default dataset and configuration, with an optional `--config` flag that accepts either the name of a stored config (under `configs/<app>/`) or an inline JSON object; an inline JSON object SHALL be merged over the app's default configuration for that run.

#### Scenario: Running a baseline evaluation
- **WHEN** a user runs the CLI's evaluate subcommand with an application name
- **THEN** the system SHALL run the corresponding `EvaluationRunner`, print the resulting `aggregate_score`, and record the run to the lineage database

#### Scenario: Evaluating with an ad-hoc configuration
- **WHEN** a user runs the CLI's evaluate subcommand with an application name and `--config '{"temperature": 0.4}'`
- **THEN** the system SHALL run the evaluation with the provided keys merged over the app's default configuration and record that merged configuration in lineage

#### Scenario: Evaluating with a stored configuration name
- **WHEN** a user runs the CLI's evaluate subcommand with an application name and `--config <name>` where `configs/<app>/<name>.json` exists
- **THEN** the system SHALL run the evaluation with that stored config merged over the app's default configuration and record the merged configuration in lineage

#### Scenario: Unknown configuration name
- **WHEN** a user runs the CLI's evaluate subcommand with `--config <name>` and no stored config with that name exists for the app
- **THEN** the CLI SHALL exit with a clear error

#### Scenario: Invalid inline configuration
- **WHEN** the `--config` value is not a JSON object or is invalid JSON
- **THEN** the CLI SHALL exit with a clear error

## ADDED Requirements

### Requirement: Evaluate with a model override

The CLI's evaluate subcommand SHALL accept an optional `--model <name>` flag that sets the model used for the app's LLM calls for that run, overriding `settings.model_name`. The model used for LLM judging SHALL remain the configured judge model, independent of `--model`.

#### Scenario: Evaluating with a specific model
- **WHEN** a user runs `refinely evaluate <app> --model <name>`
- **THEN** the run SHALL execute the app with `<name>` as the app model and SHALL record `model_name=<name>` on the run

#### Scenario: Default model used without the flag
- **WHEN** a user runs `refinely evaluate <app>` without `--model`
- **THEN** the run SHALL use `settings.model_name` as the app model and SHALL record that name on the run

### Requirement: Fan out evaluation across models

The CLI's evaluate subcommand SHALL accept an optional `--models <name1,name2,...>` flag that runs one evaluation per model in the list, recording each as a separate run with its own model name.

#### Scenario: Evaluating across multiple models
- **WHEN** a user runs `refinely evaluate <app> --models a,b,c`
- **THEN** the system SHALL produce three separate runs, one per model, each recorded with its corresponding `model_name`

#### Scenario: Empty model list
- **WHEN** a user runs `refinely evaluate <app> --models ""` or the flag value is an empty list
- **THEN** the CLI SHALL exit with a clear error

### Requirement: Manage named configs via the CLI

The CLI SHALL provide a `config` subcommand group for managing stored configs: `config save <name> --app <app> --config <json>`, `config list [--app <app>]`, `config show <name> --app <app>`, `config rm <name> --app <app>`, and `config default <app> --set <name>` / `config default <app> --clear`.

#### Scenario: Saving a config
- **WHEN** a user runs `refinely config save my-run --app extraction --config '{"temperature": 0.4}'`
- **THEN** the CLI SHALL write `configs/extraction/my-run.json` and report the written path

#### Scenario: Saving an invalid config
- **WHEN** a user runs `refinely config save <name> --app <app> --config <invalid-json>`
- **THEN** the CLI SHALL exit with a clear error and SHALL NOT create the file

#### Scenario: Setting the default config
- **WHEN** a user runs `refinely config default <app> --set <name>` and `configs/<app>/<name>.json` exists
- **THEN** the app's default config pointer SHALL be set to `<name>`

#### Scenario: Clearing the default config
- **WHEN** a user runs `refinely config default <app> --clear`
- **THEN** the app's default config pointer SHALL be cleared

#### Scenario: Showing a config
- **WHEN** a user runs `refinely config show <name> --app <app>`
- **THEN** the CLI SHALL print the JSON contents of `configs/<app>/<name>.json`

#### Scenario: Removing a config
- **WHEN** a user runs `refinely config rm <name> --app <app>`
- **THEN** the CLI SHALL delete `configs/<app>/<name>.json`
