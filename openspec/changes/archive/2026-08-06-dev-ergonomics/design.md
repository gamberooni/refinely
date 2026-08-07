# Design: dev-ergonomics

## Context

Contributing an app to Refinely today means hand-writing `apps/<name>.py` with a `register_app(AppRegistration(...))` call (six required fields), creating a dataset by hand, and adding an entry point under `[project.entry-points."refinely.apps"]` in `pyproject.toml`. There is no scaffold, no health check, and no way to inspect a dataset before running an evaluation. `load_dataset` (src/refinely/eval/datasets.py) already validates structure at load time with case-indexed `EvalError`s — so parse-level validation exists; what is missing is *inspection* (stats) and *guidance* (scaffold/doctor).

Apps live outside the refinely package (`apps/`, sibling of `src/`), are loaded via `discover_apps()` over entry points, and register via `register_app`. All existing CLI commands funnel through `cli.py`'s shared `_load_run_context(app)` helper.

## Goals / Non-Goals

**Goals:**
- Scaffold a valid `apps/<name>.py` + dataset stub in one command, with a printed entry-point hint.
- Provide a deterministic `doctor` that catches environment/schema/dataset problems with fix hints and a proper exit code.
- Provide `dataset stats <app>` for dataset inspection, plus a malformed-case report that does not raise.

**Non-Goals:**
- Editing `pyproject.toml` from the CLI (entry point wiring stays manual, by design — the scaffold only prints the line).
- Network-dependent doctor checks by default (hermetic tests must never hit the network).
- Dataset content validation beyond structural consistency (semantic checks stay in app code).
- Auto-generation of app logic (build_adapter/metrics bodies are placeholders the user fills in).

## Decisions

### D1: New `src/refinely/scaffold.py` with string templates
The scaffold writes from inline Python string templates (one for the app module, one for the dataset stub). The app template emits a `register_app` skeleton covering `build_adapter`, `metrics_factory`, `search_space`, `default_config`, `weights`, `dataset_path` with clearly-marked `TODO` placeholders and a correct module docstring + imports matching the existing app convention (`apps/extraction.py` pattern: `DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "<name>_v1.json"`). Dataset stub is `{"version": "<name>_v1", "cases": []}`.

*Why templates over a class-based generator:* the output is static, versioned with the change, and trivially reviewable; there is no runtime logic worth abstracting.

### D2: Name validation = Python identifier check
App name validation uses `str.isidentifier()` (covers the "valid Python module identifier" requirement) plus a reserved-name guard. Invalid names exit before any file write. Existing-target guard: if `apps/<name>.py` exists, error and do not overwrite.

### D3: New `src/refinely/doctor.py` — check list + exit codes
A `run_checks(settings, network: bool) -> list[CheckResult]` where each check is a small function returning `(name, ok, detail, hint)`. Checks:
1. **apps** — `discover_apps()` returns ≥1 app (already called by cli import; re-run here for a standalone result).
2. **datasets** — for each registered app, `load_dataset(registration.dataset_path)` succeeds.
3. **schema** — open `LineageDB(settings.lineage_db_path)` (enters → `init_schema`); report backfill/upgrade success.
4. **env** — `settings.openai_api_key` present (non-empty), using the existing `REFINELY_OPENAI_API_KEY is not set` semantics.
5. **network** (only with `--network`) — probe the configured `base_url` (or default OpenAI endpoint) with a lightweight call; failure is a failed check, not a crash.

Exit code: 0 if all run checks pass, 1 otherwise. Each failed check prints its hint. Rich panel rendering to match the CLI's existing style.

### D4: `dataset stats` via a new analysis function in `datasets.py`
Add `dataset_stats(path: str | Path) -> DatasetStats` (dataclass: case_count, file_size_bytes, input_field_counts: dict[str,int], expected_shape_counts, malformed: list[CaseRef]) beside `load_dataset`. It reuses `load_dataset` for parse validation (parse errors propagate as today — the CLI catches and prints the named-file/failing-case error), then computes shape summaries from the loaded cases: for each `input` key, presence count across cases; for `expected`, a coarse type histogram (dict/list/str/num) plus key presence when dict. Malformed = cases whose `input` keys deviate from the modal key set, or whose `expected` type deviates from the modal type. The report lists case ids, never raises.

### D5: CLI wiring
Three new click commands in `cli.py`, all using the existing conventions (rich panels, `_load_run_context` where app context is needed):
- `new app <name>` → `scaffold.write_app(name, dataset_path)`.
- `doctor` → `doctor.run_checks(...)` → panel + exit code.
- `dataset stats <app>` → `_load_run_context(app)` for `dataset_path`, then `dataset_stats` → panel.

## Risks / Trade-offs

- **Scaffold goes stale vs real app conventions** → Templates are small and versioned; the `apps/extraction.py` pattern is the reference and should be re-checked when the scaffold changes.
- **`doctor` schema check opens/writes the lineage DB** → Same behavior as every CLI command that touches `LineageDB`; it is the intended upgrade path (`init_schema` backfills).
- **`dataset stats` malformed logic is heuristic (modal shapes)** → It reports, never blocks; the parse-level guarantee comes from `load_dataset` which remains strict.
- **Network probe flakiness** → Opt-in only; deterministic checks are what gate the exit code, so CI/hermetic runs are unaffected.

## Migration Plan

No schema or data changes (doctor only *initializes* the existing DB). New commands and modules are purely additive.

## Open Questions

- Whether `doctor` should also validate that every registered app is importable *in a fresh interpreter* (entry-point wiring check) — currently implicit via `discover_apps()` which already imports each entry-point module at call time.
