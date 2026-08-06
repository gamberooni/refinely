# CLI Specification

Delta for `actionable-results`: the `evaluate` and `optimize` subcommands accept a `--tags <a,b>` flag at run creation, persisting tags on the recorded runs.

## MODIFIED Requirements

### Requirement: Run evaluation via CLI
The CLI SHALL support running a baseline evaluation for a single named application (`extraction`, `qa`, or `rag`) against its default dataset and configuration, with an optional `--config <json>` flag whose JSON object SHALL be merged over the app's default configuration for that run, and an optional `--tags <a,b>` flag whose comma-separated tags SHALL be persisted on the recorded run.

#### Scenario: Running a baseline evaluation
- **WHEN** a user runs the CLI's evaluate subcommand with an application name
- **THEN** the system SHALL run the corresponding `EvaluationRunner`, print the resulting `aggregate_score`, and record the run to the lineage database

#### Scenario: Evaluating with an ad-hoc configuration
- **WHEN** a user runs the CLI's evaluate subcommand with an application name and `--config '{"temperature": 0.4}'`
- **THEN** the system SHALL run the evaluation with the provided keys merged over the app's default configuration and record that merged configuration in lineage
- **WHEN** the `--config` value is not a JSON object or is invalid JSON
- **THEN** the CLI SHALL exit with a clear error

#### Scenario: Tagging an evaluation run
- **WHEN** a user runs the CLI's evaluate subcommand with an application name and `--tags candidate,prod`
- **THEN** the system SHALL record the run with the tags `candidate` and `prod` on its lineage row

### Requirement: Run optimization via CLI
The CLI SHALL support running an Optuna optimization study for a single named application, with an optional flag to override the number of trials (default 15), and an optional `--tags <a,b>` flag whose comma-separated tags SHALL be persisted on every trial run recorded from the study.

#### Scenario: Running an optimization study
- **WHEN** a user runs the CLI's optimize subcommand with an application name and an optional trials count
- **THEN** the system SHALL run the corresponding Optuna study for that many trials, printing the best trial's score and configuration at the end, and recording every trial to the lineage database

#### Scenario: Tagging an optimization study
- **WHEN** a user runs the CLI's optimize subcommand with an application name and `--tags candidate`
- **THEN** the system SHALL record every trial run from the study with the tag `candidate`
