# Proposal: DSPy compile pipeline

## Why

Refinely can tune application *configuration* (Optuna) but cannot optimize LLM *behavior* — prompts, instructions, examples, reasoning patterns. The design docs ("GenAI App Eval and Optimization", ADR-002: Use DSPy and Optuna Together) specify a second, orthogonal optimizer: DSPy compiles an app's LLM program against the same dataset and metrics, producing an optimized program artifact the app then consumes. Refinely has the dataset, Metric protocol, and lineage — but no DSPy integration.

## What Changes

- Add an optional `dspy` dependency group (`uv sync --group dspy`); all `import dspy` stays lazy so the core package and hermetic test suite never require it.
- Extend `AppRegistration` with an optional `dspy_factory(settings)` field returning a `DspyProgramSpec` (fresh `dspy.Module` builder + case→example converter + prediction→output mapper). Apps without it are unaffected and simply do not support compilation.
- Add a compile harness (`src/refinely/dspy/`): wires `dspy.LM` from refinely `Settings` (OpenAI-compatible `base_url`), builds a BootstrapFewShot optimizer whose metric is a **bridge over the app's registered metrics** (prediction → synthetic `Result` → `EvaluationRunner` → weighted aggregate), splits the app's dataset into train/val, and saves `optimized_program.json` artifacts.
- Record compiles in lineage: new `dspy_compiles` table (separate from Optuna `evaluation_runs`, per the separate-optimizers decision) with baseline vs. compiled scores.
- Add `refinely compile <app>` CLI subcommand and a `--program <path>` flag on `refinely evaluate`; apps consume the compiled artifact via an optional `program_path` on `build_adapter`.
- All three demo apps (extraction, QA, RAG) declare DSPy programs: extraction = one typed signature; QA = generation step (retrieval stays deterministic outside the program); RAG = expansion/rerank/generation modules with retrieval as a pre-pass.

## Capabilities

### New Capabilities
- `dspy-compilation`: the DspyProgramSpec contract, per-app dspy_factory registration, the compile harness (LM wiring, BootstrapFewShot, metric bridge, train/val split, artifact save/load), and runtime consumption of compiled programs via `program_path`.

### Modified Capabilities
- `cli`: new `compile` subcommand and `--program` flag on `evaluate`.
- `experiment-lineage-tracking`: new `dspy_compiles` table recording compile runs, artifacts, and baseline/compiled scores.

## Impact

- `pyproject.toml`: new dependency group `dspy`.
- `src/refinely/registry.py`: `AppRegistration.dspy_factory` optional field (backward compatible; register_app callers unaffected).
- New module `src/refinely/dspy/` (spec, bridge, compile runner); `apps/extraction.py`, `apps/qa.py`, `apps/rag.py` gain `dspy_factory` + optional `program_path` consumption.
- `src/refinely/tracking/db.py`: `dspy_compiles` schema + `record_compile`/`best_compile`.
- `src/refinely/cli.py`: compile command + evaluate flag.
- Tests: hermetic harness tests (stub dspy), metric-bridge tests with `StubLLMClient`, `pytest.importorskip("dspy")` smoke tests. Target ~100 total.
