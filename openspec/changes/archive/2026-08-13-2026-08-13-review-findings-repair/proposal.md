# Proposal: review-findings-repair

## Why

A deep review (`docs/review/deep-dive.md`) concluded that the headline feature — `optimize` — cannot currently be trusted, and that six confirmed bugs corrupt data interpretation even in the trustworthy parts of the tool. The credibility killers:

- **The LLM judge judges itself.** The judge model defaults to the generator model, and the judge prompt leaks the expected answer — so the judge re-measures what `fuzzy_match` already measures deterministically (40–70% of QA/RAG weight is double-counted). Every judge-weighted comparison, including the `--models` axis, is invalidated.
- **No statistical controls in `optimize`.** 15 TPE trials, one sample per config, no holdout, no CIs, no significance test. On the demo datasets most objective weight is near-inert; a published rule of thumb is that N=20–40 config comparisons are ~12% likely to be pure noise.
- **The compile tier is the least verified part of the stack.** The DSPy training metric scores cost/latency as constants (zeroed usage) and RAG `retrieval_recall` as a hard 0.0 during training (60% of RAG weight constant); baseline-vs-compiled is compared on 3 validation cases (70/30 of 10) with no repeats; only the weakest optimizer (BootstrapFewShot) is wired; no real dspy run exists end-to-end anywhere.

Plus six confirmed, reproducible bugs (compare mis-pairs cases by score-rank, export silently truncates tagged runs to 50, run-level metric means exclude failed cases while the aggregate counts them as 0.0, retrieval tie-break contradicts its own docstring, compiled runs get free cost/latency scores, CI lint is red), and a batch of doc/spec drift.

## What Changes

1. **Judge redesign (groundedness).** `LLMJudgeMetric` becomes a context-grounded judge: it scores the answer against the retrieved context (faithfulness + completeness) with the expected answer removed from the prompt; `--judge-model` decouples the judge model from the generator (never equal by default); judge identity (model + prompt version) is frozen in lineage; a self-consistency pass reports judge reliability. `fuzzy_match` stays as the disjoint lexical signal.
2. **Statistical honesty in `optimize`.** Seeded holdout split (30%), ≥3 repeats of finalist configs with mean±std, a significance gate (CI-overlap / paired test) before writing `opt-best.json`, honest "n.s." reporting, default trials raised 15 → 30.
3. **One metric-failure policy.** A case where a metric throws scores 0.0 for that metric in the run-level metric means AND the aggregate (currently they disagree); the "N cases errored" line stays.
4. **Four data-integrity bug fixes.** `compare --cases` pairs by case identity, not score-rank; `export` drops the silent 50-row LIMIT (tag filters apply before any paging); retrieval tie-break matches its docstring; compiled runs report real token usage (see 5) so cost is not free.
5. **Compile tier integrity.** Real token usage captured from the dspy LM for both the training metric and compiled comparisons (version-conditional fallback: when usage is unavailable, cost drops out of the compile objective and is marked `n/a` in compiled comparisons); RAG `retrieval_recall` excluded from the *training* objective only; optimizer default moves from BootstrapFewShot to **MIPROv2** with a `--optimizer` flag (bfs|mipro) and a score+feedback channel; a validation-size floor (default 5) plus ≥3 repeats and CI-overlap "n.s." reporting make baseline-vs-compiled honest; `evaluate --program` load errors become clean ClickExceptions.
6. **Doc/spec drift reconciled** (review §4 list: "247 passed" vs 244/3, CI gate claim, `dataset stats`, QA-only-judge spec contradiction — resolved by 1, `response_format`, TBD `Purpose` sections).

## Capabilities

### Modified Capabilities
- `evaluation-engine`: groundedness judge (no gold leak, robust parse, judge identity frozen), `--judge-model` decoupling, metric-failure policy = 0.0 everywhere, judge consistency reporting.
- `optimization-engine`: holdout split, repeats, significance gate + "n.s." reporting, default trials 30, `opt-best.json` written only on significance.
- `dspy-compilation`: real usage capture (with fallback), MIPROv2 default + `--optimizer` flag, score+feedback metric bridge, val-size floor, repeats + "n.s." on the compiled comparison, RAG `retrieval_recall` excluded from the compile objective.
- `lineage-cli-read-back`: `compare --cases` pairs by case identity; `export` never truncates.
- `cli`: `--judge-model` on evaluate/optimize; `--optimizer` on compile; `evaluate --program` load errors wrapped; `--min-val` on compile.
- `run-tags`: tag filters apply before any LIMIT in read-back queries.

## Impact

- `src/refinely/eval/metrics.py` — `LLMJudgeMetric` groundedness redesign (prompt, parse, rationale output); failure-policy alignment helpers.
- `src/refinely/eval/runner.py` — metric means include failed cases as 0.0 (consistent with aggregate).
- `src/refinely/llm/client.py` (if needed) — judge call reuse; `src/refinely/core/settings.py` — `judge_model` setting (defaults remain user-edited); `src/refinely/tracking/db.py` — `judge_model` + `judge_prompt_version` columns on `evaluation_runs` (guarded backfill).
- `src/refinely/cli/evaluate.py` / `optimize.py` / `compile.py` — new flags, error wrapping, gate wiring.
- `src/refinely/cli/readback.py` — `compare --cases` identity pairing; `export` pagination without truncation.
- `src/refinely/optimize/` — holdout split, repeats, significance gate.
- `src/refinely/dspy/` — LM usage wrapper (`lm.py`), bridge feedback channel (`bridge.py`), MIPROv2 wiring + val floor + repeats (`compile.py`), load-error wrapping (`load.py`).
- `apps/{extraction,qa,rag}.py` — judge factory signature (groundedness), extraction composite change (see D1), RAG `prediction_to_output` retained for evaluation, dspy objective pruning.
- Docs: README/`AGENTS.md`-adjacent claims, spec deltas under this change.
- Datasets: unchanged. New dependency: none (MIPROv2 uses Optuna, already in-tree; dspy group unchanged).
