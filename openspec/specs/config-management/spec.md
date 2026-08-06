# config-management Specification

## Purpose
TBD - created by archiving change config-loop. Update Purpose after archive.
## Requirements
### Requirement: Named configs stored as files

The system SHALL store named per-app configurations as JSON files on disk at `configs/<app>/<name>.json`, where `<app>` is a registered application name and `<name>` is the config name. Config files SHALL contain the app's configuration keys (prompts/params only) and SHALL NOT contain the model name, which is an orthogonal axis.

#### Scenario: Saving a named config
- **WHEN** a config `my-run` is saved for the `extraction` app
- **THEN** a file at `configs/extraction/my-run.json` SHALL exist containing the saved JSON configuration object

#### Scenario: Config names are per-app namespaced
- **WHEN** configs with the same name are saved for two different apps
- **THEN** they SHALL be stored in separate app directories (`configs/<app>/`) without colliding

### Requirement: Config file resolution

The `evaluate` command SHALL resolve `--config` as either a named config (a file under `configs/<app>/`) or an inline JSON object merged over the app's default configuration.

#### Scenario: Resolving a config by name
- **WHEN** a user runs `crucible evaluate <app> --config <name>` and a file `configs/<app>/<name>.json` exists
- **THEN** the run SHALL use the JSON parsed from that file merged over the app's default configuration

#### Scenario: Resolving an inline config
- **WHEN** a user runs `crucible evaluate <app> --config '{"temperature": 0.4}'`
- **THEN** the run SHALL use the inline JSON merged over the app's default configuration

#### Scenario: Unknown config name
- **WHEN** a user runs `crucible evaluate <app> --config <name>` and no file `configs/<app>/<name>.json` exists
- **THEN** the CLI SHALL exit with a clear error stating the config was not found

#### Scenario: Default config used when no --config given
- **WHEN** a user runs `crucible evaluate <app>` with no `--config`
- **THEN** the run SHALL use the app's default config (the per-app default pointer, falling back to the app's registered default configuration)

### Requirement: Per-app default config pointer

The system SHALL track a per-app default config that `evaluate` uses when no `--config` is given. The pointer SHALL be settable to a named config and clearable.

#### Scenario: Setting the default config
- **WHEN** a user sets the default config of an app to `<name>` and `<name>` exists in `configs/<app>/`
- **THEN** subsequent evaluate runs for that app without `--config` SHALL use `<name>`

#### Scenario: Setting a default that does not exist
- **WHEN** a user sets the default config of an app to `<name>` and no `configs/<app>/<name>.json` exists
- **THEN** the CLI SHALL exit with a clear error and SHALL NOT change the pointer

#### Scenario: Clearing the default config
- **WHEN** a user clears the default config of an app
- **THEN** subsequent runs without `--config` SHALL fall back to the app's registered default configuration

#### Scenario: Listing marks the default
- **WHEN** a user lists an app's configs and a default config is set
- **THEN** the list SHALL mark the default config distinctly (e.g. with a star)

### Requirement: Config lifecycle operations

The system SHALL support listing, inspecting, and removing named configs per app.

#### Scenario: Listing configs for an app
- **WHEN** a user lists configs for an app that has stored configs
- **THEN** the CLI SHALL list every config name under `configs/<app>/`

#### Scenario: Listing configs across all apps
- **WHEN** a user lists configs without an app argument
- **THEN** the CLI SHALL list each app's stored configs

#### Scenario: Showing a config's contents
- **WHEN** a user shows config `<name>` for an app
- **THEN** the CLI SHALL print the JSON contents of `configs/<app>/<name>.json`

#### Scenario: Removing a config
- **WHEN** a user removes config `<name>` for an app
- **THEN** the file `configs/<app>/<name>.json` SHALL be deleted
- **WHEN** the removed config was the app's default
- **THEN** the default pointer SHALL be cleared

