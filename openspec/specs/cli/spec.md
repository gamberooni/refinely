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

### Requirement: Show run history via CLI
The CLI SHALL support a `show` subcommand (`crucible show <app>`) that reads the lineage database and renders the app's run history, and (`crucible show <app> --run <run_id>`) per-case results for a specific run. Run ids SHALL accept a unique prefix in place of the full id.

#### Scenario: Running show for an app
- **WHEN** a user runs `crucible show <app>`
- **THEN** the CLI SHALL render a table of the app's recorded runs, newest first, with best run and best compile summaries

#### Scenario: Running show with a run id
- **WHEN** a user runs `crucible show <app> --run <run_id>` with the full run id or a unique prefix
- **THEN** the CLI SHALL render the run's per-case results, worst cases first
- **WHEN** the prefix matches more than one run
- **THEN** the CLI SHALL exit with a clear error asking for a longer prefix
- **WHEN** no run matches
- **THEN** the CLI SHALL exit with a clear error

### Requirement: Compare runs via CLI
The CLI SHALL support a `compare` subcommand (`crucible compare <app> [--baseline <run_id>]`) that renders the app's runs with per-metric deltas against a baseline run. The baseline run id SHALL accept a unique prefix in place of the full id.

#### Scenario: Running compare for an app
- **WHEN** a user runs `crucible compare <app>`
- **THEN** the CLI SHALL render the app's runs in chronological order with per-metric deltas against the previous run

### Requirement: Export runs via CLI
The CLI SHALL support an `export` subcommand (`crucible export <app> [--format csv|json] [--output FILE]`) that writes the app's runs and metric values to a file.

#### Scenario: Running export for an app
- **WHEN** a user runs `crucible export <app>`
- **THEN** the CLI SHALL write the app's runs to a CSV file and print its path

### Requirement: Rich-formatted terminal output
The CLI SHALL render command output using `rich` formatting — panels and tables — for the `evaluate`, `optimize`, `compile`, `show`, `compare`, and `export` commands, keeping the printed data content unchanged for the existing commands.

#### Scenario: Existing commands print rich-formatted output
- **WHEN** a user runs any of `evaluate`, `optimize`, `compile`, `show`, `compare`, or `export`
- **THEN** the output SHALL be rendered with rich panels/tables and SHALL contain the same data values the command previously printed as plain text

### Requirement: Evaluate with a model override

The CLI's evaluate subcommand SHALL accept an optional `--model <name>` flag that sets the model used for the app's LLM calls for that run, overriding `settings.model_name`. The model used for LLM judging SHALL remain the configured judge model, independent of `--model`.

#### Scenario: Evaluating with a specific model
- **WHEN** a user runs `crucible evaluate <app> --model <name>`
- **THEN** the run SHALL execute the app with `<name>` as the app model and SHALL record `model_name=<name>` on the run

#### Scenario: Default model used without the flag
- **WHEN** a user runs `crucible evaluate <app>` without `--model`
- **THEN** the run SHALL use `settings.model_name` as the app model and SHALL record that name on the run

### Requirement: Fan out evaluation across models

The CLI's evaluate subcommand SHALL accept an optional `--models <name1,name2,...>` flag that runs one evaluation per model in the list, recording each as a separate run with its own model name.

#### Scenario: Evaluating across multiple models
- **WHEN** a user runs `crucible evaluate <app> --models a,b,c`
- **THEN** the system SHALL produce three separate runs, one per model, each recorded with its corresponding `model_name`

#### Scenario: Empty model list
- **WHEN** a user runs `crucible evaluate <app> --models ""` or the flag value is an empty list
- **THEN** the CLI SHALL exit with a clear error

### Requirement: Manage named configs via the CLI

The CLI SHALL provide a `config` subcommand group for managing stored configs: `config save <name> --app <app> --config <json>`, `config list [--app <app>]`, `config show <name> --app <app>`, `config rm <name> --app <app>`, and `config default <app> --set <name>` / `config default <app> --clear`.

#### Scenario: Saving a config
- **WHEN** a user runs `crucible config save my-run --app extraction --config '{"temperature": 0.4}'`
- **THEN** the CLI SHALL write `configs/extraction/my-run.json` and report the written path

#### Scenario: Saving an invalid config
- **WHEN** a user runs `crucible config save <name> --app <app> --config <invalid-json>`
- **THEN** the CLI SHALL exit with a clear error and SHALL NOT create the file

#### Scenario: Setting the default config
- **WHEN** a user runs `crucible config default <app> --set <name>` and `configs/<app>/<name>.json` exists
- **THEN** the app's default config pointer SHALL be set to `<name>`

#### Scenario: Clearing the default config
- **WHEN** a user runs `crucible config default <app> --clear`
- **THEN** the app's default config pointer SHALL be cleared

#### Scenario: Showing a config
- **WHEN** a user runs `crucible config show <name> --app <app>`
- **THEN** the CLI SHALL print the JSON contents of `configs/<app>/<name>.json`

#### Scenario: Removing a config
- **WHEN** a user runs `crucible config rm <name> --app <app>`
- **THEN** the CLI SHALL delete `configs/<app>/<name>.json`

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

