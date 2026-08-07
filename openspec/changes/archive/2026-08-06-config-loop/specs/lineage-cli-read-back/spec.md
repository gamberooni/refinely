# Lineage CLI Read-Back Specification — Delta

## ADDED Requirements

### Requirement: Compare runs with model visibility

The `compare` subcommand SHALL render a model column showing each run's `model_name`, and SHALL accept a `--model <name>` flag that restricts the comparison to runs using that model.

#### Scenario: Comparing runs with models shown
- **WHEN** a user runs `refinely compare <app>`
- **THEN** the table SHALL include a model column with each run's recorded model name

#### Scenario: Filtering the comparison by model
- **WHEN** a user runs `refinely compare <app> --model <name>`
- **THEN** the table SHALL include only runs whose `model_name` matches `<name>`

#### Scenario: Filtering with no matching runs
- **WHEN** a user runs `refinely compare <app> --model <name>` and no recorded run uses that model
- **THEN** the CLI SHALL print a message stating no runs were found for that model
