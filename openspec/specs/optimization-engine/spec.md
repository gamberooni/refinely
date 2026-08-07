# Optimization Engine Specification

## Purpose

Per-app Optuna search-space definitions, objective-function construction that wraps the evaluation engine, and the single-objective TPE optimization loop. (Adapted from change `minimal-eval-optimization-prototype`.)
## Requirements
### Requirement: Per-app Optuna search space
The system SHALL define an Optuna search space per application: for the extraction app, `temperature` (float, 0.0-1.0) and `system_prompt_variant` (categorical: "strict"/"verbose"); for the QA app, `temperature` (float, 0.0-1.0), `top_k` (int, 1-5), and `system_prompt_variant` (categorical: "strict"/"verbose").

#### Scenario: Search space produces valid configs
- **WHEN** an Optuna trial samples from an application's search space
- **THEN** the resulting configuration dict SHALL contain only parameters valid for that application and SHALL be directly usable as the `config` argument to that application's `execute` method

### Requirement: Objective function wraps evaluation
The system SHALL construct, per application and dataset, an Optuna objective function that runs a full `EvaluationRunner` pass using the trial's sampled configuration and returns the run's `aggregate_score` as the value to maximize.

#### Scenario: Objective returns the evaluation's aggregate score
- **WHEN** Optuna calls the objective function with a `trial`
- **THEN** the function SHALL sample a configuration from the search space, run the evaluation via `EvaluationRunner`, and return the resulting `aggregate_score` as a float

### Requirement: Single-objective optimization loop
The system SHALL run a single-objective Optuna study per application using the `TPESampler`, executing 15 trials, with the study persisted to a SQLite storage backend shared with the experiment lineage database file.

#### Scenario: Running an optimization study
- **WHEN** an optimization run is started for an application
- **THEN** the system SHALL create or resume an Optuna `Study` with `direction="maximize"`, `sampler=TPESampler()`, and `storage="sqlite:///<lineage-db-path>"`, and SHALL execute exactly 15 `optimize` trials unless a different trial count is explicitly passed

### Requirement: Trial results recorded to lineage
Each Optuna trial's evaluation run SHALL be recorded to the experiment lineage tables with the Optuna trial number, so optimization progress is queryable independently of Optuna's own internal storage.

#### Scenario: Trial run is linked to lineage record
- **WHEN** an optimization trial completes an evaluation run
- **THEN** the corresponding `evaluation_runs` row SHALL have `optuna_trial_number` set to that trial's number

### Requirement: Mixed-type search spaces
The system SHALL support per-app Optuna search spaces that combine continuous floats, categoricals, integers, and booleans in a single config sampling function.

#### Scenario: RAG search space
- **WHEN** the optimizer samples a config for the `rag` app
- **THEN** the config SHALL contain `temperature` (float 0.0-1.0), `system_prompt_variant` (categorical strict/verbose), `retrieval_strategy` (categorical keyword/hybrid), `top_k` (int 1-6), `query_expansion` (boolean), and `rerank` (boolean)

#### Scenario: Default config shape
- **WHEN** the CLI runs a baseline evaluation of the `rag` app
- **THEN** the default config SHALL be temperature 0.0, strict variant, hybrid retrieval, top_k 3, query_expansion off, and rerank off

### Requirement: Best config auto-save

After an optimization study completes, the system SHALL write the best trial's configuration to `configs/<app>/opt-best.json` (overwriting any previous file), and the CLI SHALL report the written path.

#### Scenario: Saving the best config
- **WHEN** an `optimize` run completes with at least one successful trial
- **THEN** a file `configs/<app>/opt-best.json` SHALL exist containing the best trial's configuration as JSON

#### Scenario: Reporting the saved path
- **WHEN** an `optimize` run completes
- **THEN** the CLI output SHALL include the path `configs/<app>/opt-best.json`

#### Scenario: No successful trials
- **WHEN** an `optimize` run completes with no successful trials
- **THEN** the CLI SHALL exit with a clear error and SHALL NOT write `opt-best.json`
