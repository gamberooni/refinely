# Proposal: config-loop

## Why

The evaluate → optimize → adopt loop ends in copy-paste: `optimize` prints a best config as JSON that the user must manually re-type into `evaluate --config`, and the model under test is a single global setting (`settings.model_name`) shared by app calls and judging — so comparing "does this config beat that config *on a different model*" is structurally impossible.

## What Changes

- Add named configs as versionable JSON files under `configs/<app>/<name>.json`, managed by a new `config` subcommand group (`save`, `list`, `show`, `rm`, `default`).
- Make `evaluate`/`optimize` resolve `--config` as either a config name or inline JSON; with no `--config`, use the app's default config (a per-app `default` pointer, editable via `config default`).
- Have `optimize` auto-save its best trial's config to `configs/<app>/opt-best.json` (overwritten each run) and print the path.
- Treat the model as an orthogonal axis: add a `--model` flag to `evaluate`/`optimize` (defaulting to `settings.model_name`), record the model on every run in a new `model_name` column on `evaluation_runs`, and support `evaluate --models a,b,c` fanning out to one run per model. The judge model stays on `settings` so judging stays consistent while the app model varies.
- Surface the model in read-back: `compare` gains a model column and a `--model` filter.

Config files hold prompts/params only — never the model name.

## Capabilities

### New Capabilities
- `config-management`: named per-app config files (`configs/<app>/<name>.json`), the `config` subcommand group (save/list/show/rm/default), and name-or-inline resolution of `--config` by evaluate/optimize, including the `opt-best.json` auto-save.

### Modified Capabilities
- `cli`: evaluate/optimize gain `--model` / `--models`; evaluate/optimize resolve `--config` as name or inline JSON; the `config` subcommand group is exposed.
- `experiment-lineage-tracking`: `evaluation_runs` gains a `model_name` column; schema init upgrades existing databases by adding the column without losing rows.
- `optimization-engine`: optimize accepts `--model` and auto-saves the best trial's config to `configs/<app>/opt-best.json`.
- `lineage-cli-read-back`: compare renders a model column and filters runs by `--model`.

## Impact

- `src/crucible/cli.py` — new `config` command group; `--model`/`--models` flags on evaluate/optimize; config-name resolution in evaluate/optimize.
- `src/crucible/tracking/db.py` — `model_name` column on `evaluation_runs` + upgrade path in `init_schema`.
- `src/crucible/reporting/render.py` — model column in `runs_table`/`compare_table`.
- New module for config file storage/management (save/list/show/rm/default + resolution), likely `src/crucible/config.py` or similar.
- New `configs/` directory convention (git-versionable, per-app namespaced).
- Datasets and app registry: unchanged. No new dependencies.
