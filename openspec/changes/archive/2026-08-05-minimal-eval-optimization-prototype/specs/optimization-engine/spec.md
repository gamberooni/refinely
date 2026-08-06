## ADDED Requirements

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
