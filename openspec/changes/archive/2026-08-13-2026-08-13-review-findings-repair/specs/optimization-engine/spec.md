# Optimization Engine Specification — Delta

## MODIFIED Requirements

### Requirement: Single-objective optimization loop

The system SHALL run a single-objective Optuna study per application using the `TPESampler`, executing 30 trials by default (overridable via `--trials`), with the study persisted to a SQLite storage backend shared with the experiment lineage database file. The loop SHALL operate on a seeded holdout split: a fixed 30% of dataset cases are held out as the validation split (at least 3 cases), the TPE objective evaluates candidates on the search split only, and the validation split SHALL NOT be seen by the sampler.

#### Scenario: Running an optimization study
- **WHEN** an optimization run is started for an application
- **THEN** the system SHALL create or resume an Optuna `Study` with `direction="maximize"`, `sampler=TPESampler()`, and `storage="sqlite:///<lineage-db-path>"`, SHALL execute 30 `optimize` trials by default (unless a different count is passed), SHALL split the dataset into search/validation (30% val, ≥ 3 cases) with a deterministic seed, and SHALL print the holdout sizes

#### Scenario: The objective only sees the search split
- **WHEN** Optuna calls the objective function with a `trial`
- **THEN** the function SHALL sample a configuration from the search space and evaluate it on the search split only; the validation split SHALL be reserved for the final gate

### Requirement: Best config auto-save

After an optimization study completes, the system SHALL write the best trial's configuration to `configs/<app>/opt-best.json` only when the best candidate's validation result is statistically significant against the baseline; otherwise the CLI SHALL report "n.s." and SHALL NOT overwrite any existing `opt-best.json`.

#### Scenario: Saving the best config
- **WHEN** an `optimize` run completes and the best candidate beats the baseline significantly on the validation split
- **THEN** a file `configs/<app>/opt-best.json` SHALL exist containing the best trial's configuration as JSON, and the CLI SHALL report the written path and the gate verdict

#### Scenario: Reporting the saved path
- **WHEN** an `optimize` run completes with at least one successful trial
- **THEN** the CLI output SHALL include the path `configs/<app>/opt-best.json`

#### Scenario: No successful trials
- **WHEN** an `optimize` run completes with no successful trials
- **THEN** the CLI SHALL exit with a clear error and SHALL NOT write `opt-best.json`

#### Scenario: Improvement is not significant
- **WHEN** an `optimize` run completes and the best candidate does not beat the baseline beyond noise on the validation split
- **THEN** the CLI SHALL report the result as not significant ("n.s."), SHALL NOT overwrite an existing `opt-best.json`, and SHALL record the n.s. status in lineage

## ADDED Requirements

### Requirement: Statistical gate with repeats

The system SHALL run the final candidates (baseline and the top candidate(s)) for at least 3 repeats each on the validation split, compute mean and standard deviation per configuration, and apply a significance test (non-overlapping 95% confidence intervals or a paired test across the shared validation cases) before deciding the gate verdict.

#### Scenario: Repeats produce variance estimates
- **WHEN** the final gate runs for an optimize study
- **THEN** the baseline and each top candidate SHALL be evaluated ≥ 3 times on the validation split, and the CLI SHALL report mean±std per configuration and the gate verdict (significant / n.s.)

#### Scenario: Gate verdict recorded
- **WHEN** an optimize study completes
- **THEN** the study's lineage records SHALL include the gate verdict and the repeat statistics used for the decision
