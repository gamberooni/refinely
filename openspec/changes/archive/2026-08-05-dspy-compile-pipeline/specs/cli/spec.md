## ADDED Requirements

### Requirement: Compile application via CLI
The CLI SHALL support a `compile` subcommand that runs the DSPy compile pipeline for a single named application, with optional flags for the training-set cap, optimizer settings (`--max-rounds`, `--max-labeled-demos`, `--max-bootstrapped-demos`), artifact output directory, and lineage database path.

#### Scenario: Running a compile
- **WHEN** a user runs the CLI's compile subcommand with an application name that declares a `dspy_factory`
- **THEN** the system SHALL run the compile pipeline, print the artifact path and baseline/compiled scores, and record the compile in the lineage database

#### Scenario: Compiling an app without a program
- **WHEN** a user runs the CLI's compile subcommand with an application name that has no `dspy_factory`
- **THEN** the system SHALL exit with a clear error naming the applications that do support compilation

### Requirement: Evaluate with a compiled program
The CLI's evaluate subcommand SHALL accept an optional `--program <path>` flag that is forwarded to the application's `build_adapter` so the evaluation runs through a compiled artifact.

#### Scenario: Evaluating a compiled artifact
- **WHEN** a user runs the CLI's evaluate subcommand with an application name and `--program <path>`
- **THEN** the system SHALL build the app with that program path, run the evaluation, print the aggregate score, and record the run to the lineage database
