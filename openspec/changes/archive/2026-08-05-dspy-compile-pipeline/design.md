# Design: DSPy compile pipeline

## Context

Refinely today optimizes only application *configuration* via Optuna (TPE over per-app search spaces). Apps hard-code their LLM behavior as prompt templates (`SYSTEM_PROMPTS` dicts) and the search space can only pick among pre-authored variants. The target architecture (vault design docs, ADR-002 "Use DSPy and Optuna Together") calls for a second, separate optimizer: DSPy compiles an app's LLM program — prompts, instructions, demonstrations, reasoning — against the same dataset and metrics refinely already owns, and the application consumes the optimized component. Refinely's registry refactor made app contracts opt-in via `AppRegistration`, so the DSPy surface follows the same pattern: apps declare a program; the framework provides the harness.

## Goals / Non-Goals

**Goals:**
- Optional per-app DSPy program declaration (`dspy_factory` on `AppRegistration`) with zero impact on apps that don't declare one.
- A `refinely compile <app>` flow that trains a BootstrapFewShot optimizer on the app's own dataset using the app's own registered metrics (via a bridge, no second metric system), saves `optimized_program.json`, and records the compile in lineage.
- Runtime consumption of the compiled artifact (`refinely evaluate <app> --program <path>`) through an optional `program_path` on `build_adapter`.
- All three demo apps covered: extraction (typed signature), QA (generation step), RAG (expansion/rerank/generation modules).

**Non-Goals:**
- No MIPROv2/GEPA/etc. optimizers in this change (the harness is parameterized so adding them later is local).
- No prompt-artifact diffing or prompt registry UI; artifacts are plain files + lineage rows.
- No changes to the Optuna path beyond the shared registration surface.
- DSPy stays an optional dependency: the core package and hermetic tests never require it.

## Decisions

1. **DSPy 3.x as an optional uv group.** `dspy>=3.1` in a new `[dependency-groups] dspy`; all `import dspy` inside functions. Rationale: modern API (`BootstrapFewShot`, typed signatures, `LM(model, api_base=...)`, `module.save()/load()`), maintained, supports the OpenAI-compatible gateway via LiteLLM kwargs. Alternative considered: dspy 2.x (`teleprompt`) — legacy, rejected; framework-native reimplementation — rejected for scope.
2. **`AppRegistration.dspy_factory(settings) -> DspyProgramSpec`.** Spec dataclass in `src/refinely/dspy/spec.py`: `build() -> dspy.Module` (fresh program), `prepare_example(case) -> dspy.Example` (program inputs; deterministic retrieval happens here, outside the program), `prediction_to_output(pred) -> dict` (map prediction to the app's output shape so refinely metrics and `Result` semantics apply unchanged). Rationale: mirrors the registry's opt-in philosophy; keeps DSPy details out of core.
3. **Metric bridge over registered metrics.** `metric(gold, prediction, trace)`: build a synthetic `Result` (output from `prediction_to_output`, zero token usage/latency), reconstruct the `EvalCase`, run the app's `metrics_factory` metrics through `EvaluationRunner`'s scoring path, return the weighted aggregate. Rationale: one metric system; `LLMJudgeMetric` stays meaningful inside bootstrap (extra judge call per candidate is accepted cost).
4. **LM wiring from refinely Settings.** `dspy.LM(f"openai/{settings.model_name}", api_base=settings.base_url, api_key=settings.api_key, temperature=...)` then `dspy.configure(lm=...)`. Rationale: works with the local gateway unchanged; no new config surface. Temperature for the teacher comes from the registration's default config.
5. **Compile flow lives in `src/refinely/dspy/compile.py`.** Train/val split of the app's dataset (default 70/30 by index, `--max-examples` caps training size), `BootstrapFewShot(metric=bridge, max_bootstrapped_demos, max_labeled_demos, max_rounds)` (CLI flags), baseline eval (app default config) + compiled eval on the val set, artifact written to `--output-dir` (default `.`), lineage row recorded. Rationale: mirrors `optimize`'s structure (runner + objective + study as library calls, thin CLI).
6. **New lineage table `dspy_compiles`.** Columns: `compile_id` (uuid), `app_name`, `dataset_version`, `optimizer`, `config` (JSON), `artifact_path`, `baseline_score`, `compiled_score`, `created_at`. New methods `record_compile(...)` and `best_compile(app)`. Rationale: DSPy is a separate optimizer per the design docs; keeping compiles out of `evaluation_runs` preserves the Optuna-only contract of that table. `init_schema` creates both old and new tables idempotently.
7. **CLI: `refinely compile <app>` + `refinely evaluate <app> --program <path>`.** Compile flags: `--max-examples`, `--max-rounds`, `--max-labeled-demos`, `--max-bootstrapped-demos`, `--output-dir`, `--lineage-db`. Apps without a `dspy_factory` → click error listing which apps support compile; dspy not installed → error with install hint. `build_adapter(client, settings, program_path=None)` — optional kwarg, backward compatible; apps load the artifact via `module.load(path)` at build time and use it in `execute` when present, falling back to hardcoded prompts otherwise.
8. **Demo programs.** Extraction: `dspy.Signature` with typed fields (`text -> field_name: str, field_value: str`) as a `Predict`/`ChainOfThought` module mirroring the strict prompt. QA: `question, snippets -> answer` generation module (retrieval in `prepare_example` via the corpus). RAG: `ChainOfThought`-style modules for expansion and rerank plus a generation module over `question + snippets`, retrieval pre-pass in `prepare_example`; `prediction_to_output` reconstructs the citation-bearing output dict.

## Risks / Trade-offs

- [Compile makes real LLM calls and can be slow/costly] → capped by `--max-examples`/`--max-rounds` defaults; user-invoked command, not part of the hermetic suite.
- [BootstrapFewShot acceptance uses the LLM judge metric → extra judge calls] → accepted; bridge is metric-generic, users can pass a metric set without the judge in future.
- [DSPy 3.x API drift or version incompatibility] → `importorskip`-gated smoke tests + pinned major version; lazy imports isolate it from core.
- [Typed/structured outputs differ across dspy versions] → extraction program uses plain typed signature fields (stable across 3.x); verified during implementation.
- [Existing `evaluation_runs` consumers unaffected] → schema change is additive-only (new table), `init_schema` stays idempotent.

## Migration Plan

Additive change: new optional dependency group, new module, one new table, one new CLI command + flag. Nothing existing changes shape; `build_adapter` gains a keyword arg with a default. No data migration. Rollback = revert files; lineage DBs created with the new table work fine on older code (extra table ignored).

## Open Questions

- Whether `optimize` should also accept `--program` (deferred — YAGNI; `evaluate --program` covers artifact scoring).
