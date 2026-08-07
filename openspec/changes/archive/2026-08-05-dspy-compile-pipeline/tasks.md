# Tasks: dspy-compile-pipeline

## 1. Dependency group

- [x] 1.1 Add `[dependency-groups] dspy = ["dspy>=3.1"]` to pyproject.toml and `uv sync --group dspy`
- [x] 1.2 Add `_dspy` lazy-import helper (import inside functions only, raise `EvalError` with install hint when missing)

## 2. Contract: DspyProgramSpec + registry

- [x] 2.1 Create `src/refinely/dspy/__init__.py` + `spec.py` with `DspyProgramSpec` dataclass (`build`, `prepare_example`, `prediction_to_output`)
- [x] 2.2 Add optional `dspy_factory` field to `AppRegistration` (backward compatible) and expose it in `register_app`/`get_registration`
- [x] 2.3 Registry tests: dspy_factory roundtrip, absence OK (no breakage)

## 3. Compile harness

- [x] 3.1 `src/refinely/dspy/bridge.py`: metric bridge (prediction → synthetic Result → app's metrics via EvaluationRunner scoring → weighted aggregate)
- [x] 3.2 `src/refinely/dspy/lm.py`: `configure_lm(settings, temperature)` via `dspy.LM` with OpenAI-compatible `api_base`
- [x] 3.3 `src/refinely/dspy/compile.py`: train/val split (70/30, `--max-examples` cap), BootstrapFewShot runner, baseline + compiled eval on val, artifact save (`optimized_program.json`), returns result dataclass

## 4. Lineage

- [x] 4.1 `dspy_compiles` table in `init_schema` (idempotent, additive) + `record_compile(...)` and `best_compile(app)` in LineageDB
- [x] 4.2 Lineage tests: schema creation, record roundtrip, best_compile ordering

## 5. Demo programs

- [x] 5.1 ExtractionApp: `dspy_factory` (typed signature `text -> field_name, field_value`), runtime `program_path` consumption
- [x] 5.2 QAApp: `dspy_factory` (generation module over question+snippets, retrieval in `prepare_example`), runtime `program_path`
- [x] 5.3 RAGApp: `dspy_factory` (expansion/rerank/generation modules, retrieval pre-pass), runtime `program_path`
- [x] 5.4 All three `build_adapter`s accept optional `program_path` kwarg; `apps/__init__.py` unchanged

## 6. CLI

- [x] 6.1 `refinely compile <app>` subcommand (flags: `--max-examples`, `--max-rounds`, `--max-labeled-demos`, `--max-bootstrapped-demos`, `--output-dir`, `--lineage-db`; clear errors for no-dspy_factory and dspy-not-installed)
- [x] 6.2 `refinely evaluate --program <path>` flag forwarded to `build_adapter`
- [x] 6.3 CLI tests: compile flag plumbing (stub harness), evaluate --program, error paths

## 7. Tests + verification

- [x] 7.1 Harness unit tests with stub dspy module (monkeypatched compile flow, bridge, split)
- [x] 7.2 Metric-bridge tests with StubLLMClient (extraction/QA/RAG program specs, prediction_to_output roundtrip)
- [x] 7.3 `pytest.importorskip("dspy")` smoke tests (real dspy, no network)
- [x] 7.4 Full suite green (~100 tests) + real-API smoke `refinely compile extraction` against local gateway
- [x] 7.5 `openspec verify` → archive change → sync main specs

## 8. Docs

- [x] 8.1 AGENTS.md / CONTRIBUTING.md / README.md: dspy group, compile command, program-path consumption, test count
- [x] 8.2 docs/architecture.md + docs/overview.md: DSPy optimizer component (separate from Optuna)
- [x] 8.3 docs/integration.md: DSPy compile usage pattern (no external repo mention)
