# Optimization Engine Specification — Delta

## ADDED Requirements

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
