# Tasks: review-findings-repair

TDD ordering: regression tests red → fix → green, per group. Decisions referenced as (D#) per design.md.

## 1. Data-integrity bug fixes (review A1–A5, D3/D4)

- [x] 1.1 **compare `--cases` pairs by case identity** (D4): replace score-rank-sorted positional zip in `src/refinely/cli/readback.py` with pairing by case id; report unmatched cases. Red test: two runs whose cases share equal scores must not cross-pair.
- [x] 1.2 **export drops the silent 50-row LIMIT** (D4): remove the LIMIT from the export query in `readback.py` (paginate internally with deterministic ordering); tagged exports must be complete. Red test: >50 tagged runs export fully.
- [x] 1.3 **Metric-failure policy = 0.0 everywhere** (D3): run-level metric means in `src/refinely/eval/runner.py` include failed cases as 0.0, consistent with the aggregate; "N cases errored" rendering retained. Red test: mean == aggregate-implied mean with a throwing metric.
- [x] 1.4 **Retrieval tie-break matches its docstring** (apps/common.py scorer): align sort order with the documented intent; red test asserts the documented ordering.
- [x] 1.5 **CI lint green** (D8): fix `cli/__init__.py` E402 (or reorder imports) so the lint gate passes at HEAD; verify with the repo's lint command.

## 2. Judge redesign (D1)

- [x] 2.1 **Groundedness judge metric**: rewrite `LLMJudgeMetric` in `src/refinely/eval/metrics.py` — prompt = {question, answer, context} only (no gold), structured 0.0–1.0 score + one-line rationale, robust parse (fence stripping + repair, consistent with `chat_structured` fallback). Update `StubLLMClient`-based tests.
- [x] 2.2 **`--judge-model` decoupling** (D1): `--judge-model` flag on evaluate/optimize/compile; `settings.judge_model`; enforcement rule judge ≠ generator unless explicit override with loud warning.
- [x] 2.3 **Judge identity frozen in lineage** (D1): `judge_model` + `judge_prompt_version` columns on `evaluation_runs` via guarded `ALTER TABLE` backfill (`_backfill_columns`); recorded per run; surfaced in `show`/read-back.
- [x] 2.4 **Extraction composite deterministic-only** (D1, owner-confirmed): already exact_match 0.7/latency 0.15/cost 0.15 with no judge — no code change; spec delta + registry guard cover it.
- [x] 2.5 **Judge consistency self-check** (D1): `judge_agreement` + `evaluate --judge-consistency`; test with stub.
- [x] 2.6 **Weight re-verification**: `test_weights_are_subset_of_metric_names` regression guard in test_registry.

## 3. Statistical honesty in optimize (D2)

- [x] 3.1 **Seeded holdout split** helper (30% val, val ≥ 3 enforced) + tests (split is deterministic for a seed; val never seen by the sampler).
- [x] 3.2 **Objective evaluates on the search split only** in `src/refinely/optimize/`; baseline for the gate evaluates on val.
- [x] 3.3 **Final gate**: ≥3 repeats of baseline + top candidate(s) on val, mean±std, significance test (non-overlapping 95% CIs or paired test); tests with stub.
- [x] 3.4 **Gated `opt-best.json`** (D2): write only when significant; else "n.s." report + no overwrite; n.s. status recorded in lineage; tests for both branches.
- [x] 3.5 **Default trials 15 → 30**; CLI prints holdout sizes + gate verdict; tests updated.

## 4. Compile tier integrity (D5/D6/D7/D8)

- [x] 4.1 **Usage-capturing LM wrapper** (`src/refinely/dspy/lm.py`): `configure_lm` returns a wrapper recording per-call usage; bridge reads real usage so cost/latency are real in the training metric and final comparison; version-conditional fallback (usage unavailable → cost excluded from compile objective + marked `n/a` in compiled comparisons, noted in lineage). Tests with a fake dspy LM exposing usage.
- [x] 4.2 **RAG compile objective** (D5): exclude `retrieval_recall` from the training metric only; final `evaluate --program` keeps it with real indices. Test: RAG bridge objective weights exclude retrieval_recall.
- [x] 4.3 **MIPROv2 default + `--optimizer {bfs,mipro}`** (D6): swap default in `compile.py`, CLI flag, per-optimizer hyperparameter flags; **score+feedback channel**: `make_dspy_metric` returns `dspy.Prediction(score=aggregate, feedback=...)` from judge rationale + failing sub-metrics. Harness tests for both optimizers with fake dspy.
- [x] 4.4 **Val floor + repeats + n.s.** (D7): `--min-val` (default 5) rejection with clear message; ≥3 repeats of baseline/compiled on val, mean±std, CI-overlap → "n.s." in `CompileResult` + `dspy_compiles`; CLI never claims improvement when n.s. Tests for floor and n.s. branches.
- [x] 4.5 **`evaluate --program` load errors wrapped** (D8): missing dspy / bad artifact → ClickException with the install command; test.
- [x] 4.6 **Real-dspy integration test** (item 13): one end-to-end compile test (`test_compile_program_real_dspy_stub_lm`, runs under CI job `test-dspy` with `--group dspy`, stub LM, no network) asserting baseline-vs-compiled on a tiny synthetic case; skips when dspy absent. Also fixed the litellm pin (`<1.92` per AGENTS.md — dspy group now installs; empirically verified dspy 3.3.0 LM history/usage + Prediction metric contract).

## 5. Doc/spec drift (B, D8)

- [x] 5.1 README "247 passed" → actual count (274; CONTRIBUTING.md ×3 + docs/overview.md).
- [x] 5.2 CI gate claim reconciled (CONTRIBUTING: "no CI gate" → CI runs check/format/tests + dspy job).
- [x] 5.3 `dataset stats` claim verified/reconciled — the CLI now exposes `dataset stats <app>` as a group matching the developer-tools + cli specs (was a bare `dataset <app>` command).
- [x] 5.4 QA-only-judge spec contradiction resolved (evaluation-engine spec delta, D1).
- [x] 5.5 `response_format` drift reconciled (CONTRIBUTING + docs/integration wording: sends json_object, parsing never depends on it).
- [x] 5.6 TBD spec `Purpose` sections filled (config-management, developer-tools, run-tags).
- [x] 5.7 Spec deltas under this change applied via `openspec archive` (specs/ merged into openspec/specs/, change archived).

## 6. Verification

- [x] 6.1 Full suite green (`uv run pytest tests/ -q`) — **274 passed, 0 skipped** (dspy group installed); all new regression tests present.
- [x] 6.2 `refinely doctor` passes (no network).
- [x] 6.3 Manual smoke with StubLLMClient: evaluate (judge warning + `--judge-model`), optimize (gate n.s. + significant), compile (mipro + bfs + real-dspy stub-LM integration), compare `--cases`, export `--tag` — all covered by CLI tests.
- [x] 6.4 Verification: review's exact metric-failure repro (fair mean 0.333 vs displayed 0.5) now consistent; real dspy 3.3 `LMHistoryEntry.usage` shape extracted correctly; scope scan shows no files beyond the brief. (loom reviewer subagent timed out twice in this environment — coordinator ran the verification directly.)
