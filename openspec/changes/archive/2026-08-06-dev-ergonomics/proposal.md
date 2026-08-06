# Proposal: dev-ergonomics

## Why

Adding a new app to Crucible requires hand-writing a module with `register_app`, creating a dataset by hand, and manually wiring an entry point in `pyproject.toml` — no scaffold, no feedback, no way to inspect what you're testing against. Environment or schema problems surface only as opaque failures mid-run. The contributor loop is the least-supported part of the tool.

## What Changes

- **`crucible new app <name> [--dataset <path>]`**: scaffolds `apps/<name>.py` with a complete `register_app` skeleton (placeholders for `build_adapter`, `metrics_factory`, `search_space`, `default_config`, `weights`, `dataset_path`) and a `datasets/<name>_v1.json` stub. Prints the `[project.entry-points."crucible.apps"]` line to add, but does **not** edit `pyproject.toml`.
- **`crucible doctor`**: deterministic checks — app discovery via `discover_apps()`, datasets load (`load_dataset` on each registered app), lineage DB schema current (`LineageDB`), env API key present. Opt-in `--network` probe (no live calls by default, keeps tests hermetic). Non-zero exit code when any check fails; a fix hint per failed check.
- **`crucible dataset stats <app>`**: case count, file size, per-field input/expected shape summary, and a malformed-case report (structural issues beyond the parse failures `load_dataset` already rejects with case-indexed errors).

## Capabilities

### New Capabilities
- `developer-tools`: scaffolding a new app module + dataset stub, health checks via `doctor`, and dataset inspection via `dataset stats`.

### Modified Capabilities
- `cli`: adds the `new app`, `doctor`, and `dataset stats` subcommands (new CLI surface, existing commands unchanged).

## Impact

- `src/crucible/cli.py` — three new subcommands: `new app`, `doctor`, `dataset stats`.
- New `src/crucible/scaffold.py` — app module + dataset stub templates.
- New `src/crucible/doctor.py` — deterministic check runner (discovery/dataset/schema/env) + optional network probe; fix hints; exit codes.
- `src/crucible/eval/datasets.py` — add a stats/analysis function (case count, sizes, field shapes, malformed-case report); existing `load_dataset` validation is unchanged.
- No new dependencies; network probing stays opt-in to preserve hermetic tests.
