## Context

`docs/` in this repo describes a full enterprise "AI Application Evaluation & Optimization Platform" (Evaluation Engine + Metrics + DSPy + Optuna + MLflow + multi-tenant governance, per 12 design docs and 12 accepted ADRs). Rather than build that platform, we're validating its three riskiest architectural bets cheaply, in a single small Python package:

1. Does a black-box `ApplicationAdapter.execute(input, config)` contract cleanly decouple genuinely different toy applications from a shared evaluation harness? (docs 02/05, ADR-001/010)
2. Does wrapping evaluation as an Optuna objective function actually improve an aggregate score over trials? (docs 04/06/07, ADR-002/007)
3. Does lightweight SQLite-based lineage tracking give useful, queryable reproducibility without MLflow? (docs 03/12, ADR-009)

This is a from-scratch prototype in an otherwise-empty `src/refinely/` package (currently only `__init__.py` + `py.typed`). No existing specs, no git history, no production constraints.

## Goals / Non-Goals

**Goals:**
- Two toy applications with meaningfully different config/execution shapes (structured extraction vs. retrieval-lite QA) behind one shared adapter protocol.
- Real OpenAI calls (no mocked/deterministic LLM logic) so Optuna trials produce genuinely different scores.
- A minimal but real evaluation loop: dataset → per-case execute → per-case metrics → weighted aggregate score.
- A minimal but real optimization loop: Optuna `TPESampler`, single objective, 15 trials/app, wrapping the evaluation loop as the objective function.
- Full experiment lineage (app + dataset version + config + metrics) queryable from a local SQLite file, with per-case detail for debugging.
- A `click` CLI to drive both loops.
- Unit-testable plumbing (dataset loading, metrics, aggregation, lineage read/write, Optuna wiring) via a stub `LLMClient`, with zero live-API calls in the test suite.

**Non-Goals:**
- No DSPy prompt optimization (Optuna alone covers the eval↔optimize loop under test; DSPy is a separate, un-tested bet).
- No MLflow (SQLite substitutes as the lightweight tracking backend for this prototype).
- No multi-tenancy, RBAC, audit trail, or governance (docs 10) — single local user, single machine.
- No async task queue, worker pool, Kubernetes, or Temporal (docs 09) — synchronous/async-in-process execution only.
- No OpenTelemetry tracing, dashboards, or alerting (doc 08).
- No multi-objective/Pareto optimization — single scalar weighted objective only.
- No production-traffic optimization (ADR-012 still applies in spirit, though there's no production system here).

## Decisions

**LLM client: adapt user-provided `AsyncOpenAIClient` pattern almost verbatim.**
Reuses a battle-tested pattern (JSON-schema-forced structured output, fence-stripping + repair-retry on malformed JSON, `tenacity` retry on transient errors, per-model temperature support detection) rather than reinventing an OpenAI wrapper, paired with `refinely.core.exceptions.LLMError`. Alternative considered: use `litellm` for multi-provider abstraction — rejected because this prototype only needs OpenAI and the existing client pattern is already proven.

**Two apps chosen for adapter-shape diversity, not domain realism.**
`ExtractionApp` (structured JSON output via `chat_structured`) and `QAApp` (in-memory keyword/substring retrieval + `chat_text`/`chat_structured` answer) were chosen specifically because their config shapes and execution paths differ (no config knob overlap except `temperature`/`system_prompt_variant`). This directly tests whether `ApplicationAdapter.execute(input, config)` is sufficient as a uniform contract (ADR-001/010) without the evaluation engine needing to know which app it's running.

**No real vector DB for QA retrieval.**
In-memory keyword/substring snippet matching is sufficient to exercise the `top_k` config knob and produce genuinely different LLM inputs per trial; a real vector store would add infra weight without changing what's being tested (the adapter/eval/optimize loop, not retrieval quality).

**Metrics: deterministic-first, one LLM-judge as a deliberate exception.**
`ExactMatchMetric`/`FuzzyMatchMetric`/`LatencyMetric`/`CostMetric` are cheap and deterministic, keeping the evaluation loop fast and Optuna trials cheap. `LLMJudgeMetric` is included only for QA (not extraction) specifically to validate that non-deterministic, LLM-based metrics can be folded into the same weighted-aggregation pattern (doc 06) without special-casing the runner.

**Optuna: single study per app, SQLite storage shared with lineage DB.**
Using one SQLite file for both Optuna's internal trial bookkeeping and the custom lineage tables (`evaluation_runs`/`metric_results`/`case_results`) avoids running two separate storage mechanisms. Alternative considered: separate `optuna_study.db` + `lineage.db` — rejected as unnecessary complexity for a single-machine prototype; a single file is simpler to inspect and back up. `TPESampler` (Optuna's default Bayesian-style sampler) is used over `RandomSampler` because part of what's being validated is whether *directed* search improves scores over trials, not just whether any search does.

**Lineage: three tables, not two.**
`evaluation_runs` + `metric_results` alone would satisfy ADR-009 (lineage: app+dataset+config+metrics), but a `case_results` table was added specifically for debugging — to answer "why did this config score low?" without re-running the evaluation. This trades a bit of storage for significantly better developer experience while validating the optimization loop.

**Configuration via `pydantic-settings`, not raw env vars.**
Centralizes API key, model name, and DB path in one typed `BaseSettings` class (`refinely.core.settings`) instead of scattering `os.environ.get(...)` calls, and avoids adding `python-dotenv` as a separate dependency (pydantic-settings handles `.env` loading natively).

**CLI: `click` over `argparse`.**
User explicitly requested `click` for its subcommand ergonomics (`refinely evaluate extraction`, `refinely optimize qa --trials 20`) despite `argparse` being stdlib-only and lighter weight — the ergonomics win for a CLI with multiple subcommands and options outweighs the extra dependency here.

**Module layout: package-based, mirroring the reference client's conventions.**
`core/`, `llm/`, `apps/`, `datasets/`, `eval/`, `optimize/`, `tracking/` as separate subpackages (vs. flat top-level modules) keeps each concern isolated and testable independently, and mirrors the naming style of the reference LLM client code being adapted.

**Testing: stub `LLMClient`, no live-API tests.**
Unit tests cover all non-LLM plumbing (dataset loading, metric math, aggregation, SQLite read/write, Optuna search-space/objective wiring) using a fake implementing the `LLMClient` protocol with canned responses. This keeps the test suite fast, deterministic, and runnable in CI without an API key, at the cost of not testing the real OpenAI integration automatically — acceptable since this is a manually-run prototype, not a shipped product.

## Risks / Trade-offs

- **[Risk] Real OpenAI calls make evaluation/optimization runs slow and nondeterministic across repeated runs.** → Mitigation: `seed` and `temperature=0.0` supported by the client for cases where determinism matters; 15 trials/app keeps total run time and cost bounded; not tested in the automated test suite (stubbed there).
- **[Risk] LLM-judge metric introduces its own LLM-call variance into the QA app's optimization signal (judging the judge).** → Mitigation: scoped to QA only, called with low temperature, and treated as one weighted component alongside deterministic fuzzy-match rather than the sole signal.
- **[Risk] Sharing one SQLite file between Optuna's internal tables and custom lineage tables could create naming collisions or lock contention.** → Mitigation: custom tables use distinct names (`evaluation_runs`, `metric_results`, `case_results`) unlikely to collide with Optuna's internal schema; single-process, single-machine use means lock contention is a non-issue at this scale.
- **[Risk] No live-API test coverage means client-adaptation bugs (e.g. broken JSON repair path) could go unnoticed until a real run.** → Mitigation: acceptable for a prototype; manual CLI runs during development serve as the integration check.
- **[Risk] Weighted-aggregation weights are not rigorously derived (chosen by rough intuition, e.g. correctness ~0.7, latency/cost ~0.15 each).** → Mitigation: fine for validating that the *mechanism* works; exact weight tuning is out of scope and can be revisited if the prototype graduates beyond validation.

## Migration Plan

Not applicable — this is a net-new, standalone addition to an empty package with no existing users, deployments, or data to migrate. No rollback plan needed beyond `git revert`/deleting the new `src/refinely/*` modules.

## Open Questions

- Exact weighted-aggregation weight values per app (deferred as an implementation detail; any reasonable weighting demonstrates the mechanism).
- Exact CLI flag/subcommand names beyond the two subcommands described (`evaluate <app>`, `optimize <app> [--trials N]`) — left to implementation.
- Exact `pydantic-settings` field list beyond API key / model name / db path — left to implementation.
