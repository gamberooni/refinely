# Run Tags Specification — Delta

## MODIFIED Requirements

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
