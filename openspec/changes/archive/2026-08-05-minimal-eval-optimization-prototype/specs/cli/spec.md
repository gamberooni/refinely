## ADDED Requirements

### Requirement: CLI entrypoint
The system SHALL expose a `click`-based command-line interface installed as the `crucible` console script (`[project.scripts] crucible = "crucible.cli:main"`), allowing evaluations and optimizations to be run without writing Python code.

#### Scenario: Invoking the CLI
- **WHEN** a user runs `crucible --help` after installing the package
- **THEN** the CLI SHALL list available subcommands for running an evaluation and running an optimization

### Requirement: Run evaluation via CLI
The CLI SHALL support running a baseline evaluation for a single named application (`extraction` or `qa`) against its default dataset and configuration.

#### Scenario: Running a baseline evaluation
- **WHEN** a user runs the CLI's evaluate subcommand with an application name
- **THEN** the system SHALL run the corresponding `EvaluationRunner`, print the resulting `aggregate_score`, and record the run to the lineage database

### Requirement: Run optimization via CLI
The CLI SHALL support running an Optuna optimization study for a single named application, with an optional flag to override the number of trials (default 15).

#### Scenario: Running an optimization study
- **WHEN** a user runs the CLI's optimize subcommand with an application name and an optional trials count
- **THEN** the system SHALL run the corresponding Optuna study for that many trials, printing the best trial's score and configuration at the end, and recording every trial to the lineage database
