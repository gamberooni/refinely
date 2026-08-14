# run-tags Specification

## Purpose

Run tags: recording optional comma-separated tags on run-creating commands (`evaluate`, `optimize`), filtering read-back by tag, and tag immutability. (Adapted from change `actionable-results`.)
## Requirements
### Requirement: Tag a run at creation
The system SHALL accept an optional comma-separated `--tags` list on run-creating commands (`evaluate`, `optimize`) and SHALL persist the tags on the run's lineage record. The absence of `--tags` SHALL persist a run with no tags.

#### Scenario: Tagging an evaluation run
- **WHEN** a user runs the evaluate subcommand with `--tags candidate,prod`
- **THEN** the system SHALL record the run with tags `candidate` and `prod` on its lineage row

#### Scenario: Untagged run
- **WHEN** a user runs the evaluate subcommand without a `--tags` flag
- **THEN** the system SHALL record the run with no tags

#### Scenario: Single tag
- **WHEN** a user runs the optimize subcommand with `--tags candidate`
- **THEN** the system SHALL record every trial run from that study with the single tag `candidate`

### Requirement: Filter run read-back by tag

The system SHALL accept a `--tag` filter on the run read-back commands (`show`, `compare`, `export`) and SHALL restrict the returned runs to those bearing that tag. When no run bears the tag, the command SHALL report that no matching runs were found. The tag filter SHALL be applied to the full matching set before any row limit or pagination, so filtered read-back is never silently truncated.

#### Scenario: Showing only tagged runs
- **WHEN** a user runs `refinely show <app> --tag candidate`
- **THEN** the CLI SHALL render only the runs tagged `candidate`

#### Scenario: Exporting only tagged runs
- **WHEN** a user runs `refinely export <app> --tag prod`
- **THEN** the CLI SHALL write the export containing only the runs tagged `prod`, with no truncation of the tagged set

#### Scenario: Tag filter matches nothing
- **WHEN** a user runs `refinely show <app> --tag <tag>` and no run of that app bears the tag
- **THEN** the CLI SHALL print a message stating no runs were found matching the tag

### Requirement: Tags are immutable
The system SHALL NOT provide a mechanism to re-tag an existing run after it has been recorded. Tags are fixed at run creation.

#### Scenario: No retagging command
- **WHEN** a user inspects the CLI's available commands
- **THEN** the CLI SHALL expose no subcommand or flag for modifying an existing run's tags

