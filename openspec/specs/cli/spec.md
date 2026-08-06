# CLI Specification

## Purpose

The `crucible` command-line entrypoint for running evaluations and optimizations against either toy application, allowing both workflows without writing Python code. (Adapted from change `minimal-eval-optimization-prototype`.)
## Requirements
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

### Requirement: RAG app in the CLI registry
The system SHALL register the `rag` app in the CLI so its dataset can be evaluated and optimized with the same commands as existing apps.

#### Scenario: Evaluate and optimize rag
- **WHEN** a user runs `crucible evaluate rag` or `crucible optimize rag`
- **THEN** the CLI SHALL resolve the rag dataset and corpus, build the RAG app with the shared client, and run the standard evaluation or optimization flow, recording the run in lineage

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
