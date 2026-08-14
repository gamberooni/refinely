# Design: review-findings-repair

## Context

The deep review (docs/review/deep-dive.md) verified six bugs at code level and established that the evaluation/optimization numbers cannot be trusted as shipped: the judge defaults to the generator model and leaks the expected answer into its prompt (`LLMJudgeMetric(model=settings.model_name)`, judge prompt containing the gold answer), `optimize` runs 15 TPE trials with 1 sample and no holdout on near-flat demo surfaces, the compile tier trains on a 25–60% constant objective (`dspy/bridge.py` zeroes usage/latency; RAG `prediction_to_output` returns `retrieved_indices: []`), and baseline-vs-compiled compares on `n_val=3`. The owner approved the full repair scope (brief: `.task-brief.md`).

Design forks were resolved with the owner in the interview: groundedness judge (fork 1), holdout+repeats+significance gate (fork 2), failure policy 0.0-everywhere (fork 3), capture-real-usage for the compile tier (fork 4), MIPROv2 default with `--optimizer` flag (fork 5), val floor + repeats + n.s. (fork 6).

## Goals / Non-Goals

**Goals:**
- Evaluation and optimization numbers are honest: no self-judging judge, no gold leak, no double-count, one failure policy, statistical controls in `optimize`.
- The compile tier's objective and comparisons are signal-honest, verified end-to-end, and use an instruction-level optimizer (MIPROv2) behind a `--optimizer` flag.
- All six confirmed bugs fixed with regression tests; specs and docs reconciled.

**Non-Goals (follow-ups, explicitly out of this change):**
- Structural re-layering (review Should #6: neutral models module, framework metrics into `refinely.eval.metrics`, explicit weights).
- Objective-surface re-budgeting / model-aware cost pricing (Should #7).
- Demo-trap dataset engineering — new datasets/spaces where config choices provably move scores (Should #8).
- Scaffold-trap registry check (Should #9).
- Could items (significance feature in `compare`, bring-your-own reflection LM, value-level extraction matching).

## Decisions

### D1: Groundedness judge, disjoint from fuzzy_match; extraction drops the LLM judge

`LLMJudgeMetric` is redesigned as a context-grounded judge: the prompt receives `{question, answer, context}` (the retrieved snippet block for QA/RAG) and asks for (a) faithfulness — is every claim in the answer supported by the context? — and (b) completeness — does the answer cover the context-relevant parts of the question? — returned as a structured score (0.0–1.0) plus a one-line rationale. The expected answer is **never** in the prompt. This is disjoint from `fuzzy_match` (lexical overlap with gold): the two measure different things and can coexist without double-counting.

The judge model is decoupled via `--judge-model` (evaluate/optimize/compile) backed by a new `settings.judge_model`; the default enforcement rule is **judge model ≠ generator model unless explicitly forced** (`--judge-model <same>` requires an explicit override; the CLI warns loudly otherwise). Judge identity is frozen per run: `evaluation_runs` gains `judge_model` and `judge_prompt_version` columns (guarded `ALTER TABLE` backfill, consistent with the existing schema-upgrade pattern).

**Extraction composite:** the interview accepted a "rubric variant" for extraction, but a rubric judge without a gold answer is uninformative for field extraction (plausibility ≠ correctness) and a gold-referenced rubric either double-counts `exact_match` or replaces a deterministic metric with a noisy one. Decision (owner-confirmed): extraction's composite is **deterministic-only** — `exact_match 0.7, latency 0.15, cost 0.15`, no LLM judge. (Rejected alternative on record: gold-referenced rubric replacing exact_match for near-miss credit, per the review's "if the gold must be used, drop fuzzy" rule — adds judge noise + cost to a deterministic task at demo scale.)

Judge consistency reporting: a `--judge-consistency`-style self-check (or `doctor --network` extension) re-scores a sample of cases with two judge calls at nonzero temperature and reports agreement rate, so judge noise is visible rather than hidden. A hand-labeled golden set is deferred (needs owner data).

### D2: optimize — holdout + repeats + significance gate

- Seeded (fixed seed, configurable) **30% holdout**: cases split into search/val; the TPE objective evaluates on the *search* split only; all final comparisons happen on the val split, which is never seen by the sampler. Val size must be ≥ 3; with 10-case demo datasets the split is 7/3.
- Search trials stay single-shot (cost bound); the **final gate** runs the baseline config and the top candidate(s) ≥ **3 repeats each on the val split**, producing mean±std per config.
- **Significance gate:** `opt-best.json` is written only when the best candidate's val mean beats the baseline beyond noise (non-overlapping 95% CIs or a paired test across the shared val cases). When not significant, the CLI reports **"n.s."**, does NOT overwrite an existing `opt-best.json`, and records the n.s. status in lineage.
- Default trials 15 → **30** (`--trials` still overrides); the study prints the holdout sizes and the gate verdict.
- This is honest at demo scale: on a near-flat surface the gate will usually say n.s. — that is the point (the review's "make the flat surface visible instead of fake").

### D3: Metric-failure policy — 0.0 everywhere

A case where a metric throws scores 0.0 for that metric in the run-level metric means AND in the weighted aggregate (currently the means exclude failed cases). The run result retains the existing "N cases errored" rendering. No policy flag; one consistent behavior, verified by a test asserting mean == aggregate-implied mean.

### D4: compare --cases pairs by identity; export never truncates

`compare --cases` pairs case results by **case identity** (case id / dataset position), not by score-rank order (the current sort-by-score + positional zip mis-pairs cases with equal scores). Cases present in one run but not the other are reported as unmatched; the existing dataset-version warning stays. `export` removes the silent 50-row LIMIT (paginate internally, deterministic ordering) so tagged exports are complete; tag filters apply **before** any paging (bug b).

### D5: Compile tier — real usage capture with a documented fallback

`configure_lm` installs a usage-tracking wrapper around the dspy LM (`refinely.dspy.lm`) that records the last call's token usage and latency; the metric bridge (`make_dspy_metric`) reads them so cost and latency are scored with **real numbers** in the training objective, and the compiled branches (`CompiledProgramAdapter`, QA/RAG compiled paths) read them for the final comparison. Version-conditional fallback: when the dspy version/provider does not surface usage, (a) the training objective drops cost and latency weights (quality-only), and (b) compiled runs carry `token_usage=None`, the cost/latency metrics raise `MetricUnavailableError`, and the runner **excludes** those metrics from the case scores and run means (never a fake 0.0 or 1.0) — the aggregate contribution is 0 and read-back shows the metric as missing (n/a).

**RAG objective:** the compiled RAG program does not retrieve, so `retrieval_recall` is excluded from the **compile objective only** (training metric = fuzzy + judge + citation + real latency/cost); the final `evaluate --program` comparison keeps `retrieval_recall` with real indices from the app-level retrieval (unchanged behavior, now with real usage).

### D6: Optimizer — MIPROv2 default, --optimizer flag, score+feedback channel

`compile` defaults to **MIPROv2** (dspy-native; uses Optuna, already a refinely dependency — no new dependency) behind a `--optimizer {bfs,mipro}` flag (bfs retained for compatibility/quick checks; GEPA/TextGrad are future flags). The metric bridge becomes a score+feedback channel: `make_dspy_metric` returns `dspy.Prediction(score=aggregate, feedback=...)` assembled from the judge's rationale plus the failing sub-metrics (the groundedness judge already emits a rationale — D1 — which fixes judge calibration for optimization at the same time). Compile hyperparameters stay surfaced as CLI flags per optimizer.

### D7: Compile comparison — val floor + repeats + n.s.

`compile` enforces a validation-size floor (default **5**, `--min-val` overridable): datasets too small are rejected with a clear message stating the needed minimum. Baseline and compiled each run **≥ 3 repeats on val** (mean±std); when the CIs overlap, the result and lineage record **"n.s."** and the CLI does not claim improvement. `CompileResult`/`dspy_compiles` carry the repeat stats and the n.s. flag.

### D8: CLI hygiene

`evaluate --program` wraps load failures (missing dspy, bad artifact, corrupt JSON) in a clean ClickException instead of an uncaught `EvalError` traceback. Doc/spec drift reconciled per the review §4 list; CI lint gate fixed (bug A6).

### D9: Scope — brief A–D only

The worktree brief scopes this change to the six bugs + doc/spec drift + trust architecture + compile tier (review §8 items 1–5, 10, 11–15). Should #6–9 and Could items are explicit follow-ups (see Non-Goals) so the diff stays reviewable.

## Rejected Alternatives

- **Gold-referenced rubric judge replacing fuzzy_match** (review's "drop fuzzy" rule) — rejected for QA/RAG: groundedness preserves a correctness signal without the leak and keeps the deterministic lexical metric. Reconsidered only for extraction (see D1 flag).
- **Judge self-consistency via hand-labeled golden set** — deferred: requires owner-authored labels; a self-consistency pass gives immediate visibility with no new data.
- **Repeats for every search trial** — cost-prohibitive (30 trials × 3 × LLM calls); repeats on finalists only, which is where the claim is made.
- **K-fold cross-validation for optimize** — multiplies cost and complicates the opt-best story at 10 cases; holdout+gate is honest at this scale.
- **GEPA as the first optimizer** — heavier integration and a newer dspy requirement; MIPROv2 is the least demanding instruction-level step (Optuna already in-tree), with GEPA as the next flag.
- **Keeping zero usage + re-weighting cost to 0** — equivalent to the fallback but silently; an explicit capture-with-fallback is honest about what was measured.
