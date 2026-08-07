# Refinely — Using Refinely in Your Codebase

Refinely is a framework: the evaluation and optimization machinery is generic, and
your application plugs into it. There are two integration styles:

1. **Registry app** — your module calls `register_app` and is declared as an entry
   point in group `refinely.apps`; the `refinely` CLI discovers it at startup. You
   get the `refinely` CLI subcommands and all optimize defaults for free. This
   works both for apps that ship with refinely and for external codebases.
2. **Library driver** — your codebase depends on refinely as an editable package
   and drives it from your own script, passing your app, metrics, search space,
   and weights explicitly. Best for real codebases that keep their app and data
   in their own repository and don't want refinely's CLI.

Both styles share the same core contract: an app is any object with
`execute(input: dict, config: dict) -> Result` (duck-typed, no protocol class —
`Result` comes from `refinely.llm.usage`), a config is a plain dict of
parameter values, a dataset is a list of `EvalCase`, and a metric is a `Metric`
(`evaluate(case, output) -> MetricResult`).

## Style 1: register an app

Create a module — inside the `apps/` directory (sibling of `src/`, outside the refinely package) or anywhere in your own codebase —
that calls `register_app` at import time:

```python
from pathlib import Path

from refinely.core.settings import Settings
from refinely.eval.metrics import CostMetric, FuzzyMatchMetric, LatencyMetric, Metric
from refinely.llm.client import LLMClient
from refinely.registry import AppRegistration, register_app

def sample_myapp_config(trial):
    return {
        "temperature": trial.suggest_float("temperature", 0.0, 1.0),
        "top_k": trial.suggest_int("top_k", 1, 5),
        "mode": trial.suggest_categorical("mode", ["fast", "deep"]),
    }

MYAPP_DEFAULT_CONFIG = {"temperature": 0.0, "top_k": 3, "mode": "fast"}
MYAPP_WEIGHTS = {"fuzzy_match": 0.4, "latency": 0.3, "cost": 0.3}

def _metrics_factory(client: LLMClient, settings: Settings) -> list[Metric]:
    return [FuzzyMatchMetric(), LatencyMetric(), CostMetric()]

register_app(AppRegistration(
    name="myapp",
    build_adapter=lambda client, settings: MyApp(client, settings),
    metrics_factory=_metrics_factory,
    search_space=sample_myapp_config,
    default_config=MYAPP_DEFAULT_CONFIG,
    weights=MYAPP_WEIGHTS,
    dataset_path=Path(__file__).resolve().parents[3] / "datasets" / "myapp_v1.json",
))
```

Then declare the module as an entry point in group `refinely.apps` (the value is
the module path; importing it calls `register_app`):

```toml
[project.entry-points."refinely.apps"]
myapp = "mycodebase.apps.myapp"    # or "refinely.apps.myapp" for in-tree apps
```

After `uv sync`, `discover_apps()` — which the `refinely` CLI calls at startup —
loads every entry point and the app appears everywhere: `registered_apps()` lists
it, `refinely evaluate myapp` / `refinely optimize myapp` accept it, and
`build_objective` resolves metrics, search space, and weights from the
registration when you don't override them.

To bootstrap the module instead of hand-writing it, run `refinely new app
myapp`: it scaffolds `apps/myapp.py` (a complete `register_app` skeleton) plus a
`datasets/myapp_v1.json` stub, and prints the `[project.entry-points."refinely.apps"]`
line to add — it never edits `pyproject.toml` itself.

## Style 2: refinely as a library (driver)

### 1. Depend on refinely

Add an editable path dependency from your project (or any install method that
puts `refinely` on the path — it is a plain importable package):

```bash
uv add --editable /path/to/refinely
uv sync --group dev   # optuna + SQLite deps if your project doesn't install extras
```

Your own codebase keeps everything else — app, data, runtime config — at home.

### 2. Wrap your app in a `Result`-returning class

```python
import time
from refinely.llm.usage import Result, TokenUsage

class MyApp:
    def __init__(self, pipeline, settings):
        self._pipeline = pipeline
        self._settings = settings

    def execute(self, input: dict, config: dict) -> Result:
        # Map config keys onto your own runtime config. Refinely never
        # inspects your settings — this mapping is entirely yours.
        self._settings.temperature = config["temperature"]
        self._settings.top_k = config["top_k"]
        started = time.perf_counter()
        output, usage = self._pipeline.run(input)      # your code
        return Result(
            output=output,
            token_usage=TokenUsage(prompt_tokens=usage["prompt"],
                                   completion_tokens=usage["completion"]),
            latency_seconds=time.perf_counter() - started,
        )
```

The public API is synchronous. If your pipeline is async, wrap it with
`asyncio.run(...)` — and if one case performs several LLM calls, run the whole
pipeline inside a single `asyncio.run` and aggregate token usage and latency
across the calls, so the recorded `Result` reflects the full execution.

### 3. Define a search space and defaults

```python
import optuna

def sample_myapp_config(trial: optuna.Trial) -> dict:
    return {
        "temperature": trial.suggest_float("temperature", 0.0, 1.0),
        "top_k": trial.suggest_int("top_k", 1, 5),
        "mode": trial.suggest_categorical("mode", ["fast", "deep"]),
    }

DEFAULT_CONFIG = {"temperature": 0.0, "top_k": 3, "mode": "fast"}
WEIGHTS = {"fuzzy_match": 0.4, "latency": 0.3, "cost": 0.3}   # must sum to 1.0
```

### 4. Build metrics

Reuse the generic metrics shipped with refinely, or implement your own:

```python
from refinely.eval.metrics import (CostMetric, LatencyMetric, LLMJudgeMetric,
                                   FuzzyMatchMetric, Metric, MetricResult)

class MyPrecisionMetric(Metric):
    """Deterministic app-specific metric using your own scoring code."""

    name = "my_precision"   # must match the weight-scheme key

    def evaluate(self, case, output) -> MetricResult:
        precision = compute_precision(case.expected, output)   # your code
        return MetricResult(metric_name=self.name, value=precision)
        # Note: keyword args — MetricResult is a pydantic model.
```

- `LLMJudgeMetric(client, model)` runs an LLM judge against `case.expected`; any
  object satisfying the `LLMClient` protocol (e.g. `AsyncOpenAIClient` with your
  own key/base URL) works.
- `LatencyMetric()` and `CostMetric()` read `Result.latency_seconds` /
  `token_usage` — construct them without arguments and pass no config.
- A metric that throws scores 0.0 for that case and never aborts the run.

### 5. Load or build cases

```python
from refinely.eval.datasets import EvalCase, load_dataset

cases = load_dataset("datasets/myapp_v1.json")          # versioned JSON file
# or build in code:
cases = [EvalCase(id="c1", input={"question": "..."}, expected={"answer": "..."})]
```

`case.expected` is any JSON-serializable value (a dict is fine — e.g. with
`source_indices` for retrieval-style checks). If your metric reads `case.expected`
as text, use the framework's `_expected_text` helper, which unwraps dict
expectations safely.

### 6. Wire the objective and run the study

```python
from refinely.optimize.objective import build_objective
from refinely.optimize.study import run_study

objective = build_objective(
    app_name="myapp",
    app=app,                       # your app object (duck-typed execute(input, config) -> Result)
    dataset=cases,
    dataset_version="myapp_v1",
    lineage_db_path=lineage_path,  # shared SQLite file (lineage + Optuna tables)
    client=judge,                  # LLMClient used by the LLM judge metric
    metrics=metrics,               # explicit override — defaults come from the registry
    search_space=sample_myapp_config,
    weights=WEIGHTS,
)
study = run_study("myapp", objective, lineage_path, n_trials=args.trials)
best = study.best_trial
print(f"best trial #{best.number}: aggregate_score = {best.value:.4f}")
print(f"best config: {best.params}")
```

`run_study` uses Optuna's TPE sampler, maximizes the aggregate score, persists to
`sqlite:///<lineage_path>`, and resumes an existing study (`load_if_exists=True`,
study name `refinely_{app}`) — re-running continues the same study and trial
numbers keep counting up.

### 7. Query lineage

```python
from refinely.tracking.db import LineageDB

db = LineageDB(lineage_path)
db.init_schema()
best = db.best_run("myapp")                  # highest-scoring run + parsed config
worst_cases = db.case_results_for_run(best["run_id"])   # worst-first per case
print(db.count_runs("myapp"))
```

`CaseRecord` entries from `case_results_for_run` carry an `error` field (None for
clean cases) — per-case failures are persisted, not just scored as 0.0.
`list_runs(app, limit, offset, model_name=..., tag=...)` filters by the
orthogonal model axis or by a creation-time tag.

### 8. Make it runnable

Wrap it in an argparse entry point and run with your project's tooling:

```bash
uv run python -m myapp.optimize_driver --pairs 20 --trials 3
```

## Named configs and the model axis (CLI)

The `refinely` CLI adds two conveniences on top of the library core:

- **Named configs.** `refinely config save my-run --app myapp --config
  '{"temperature": 0.4}'` writes `configs/myapp/my-run.json`. `--config` on
  `evaluate` then accepts that name or an inline JSON object; with no `--config`,
  `evaluate` uses a per-app default pointer (`refinely config default myapp --set
  my-run`) or the app's registered default config. `optimize` auto-saves the best
  trial's config to `configs/myapp/opt-best.json`.
- **The model is an orthogonal axis.** `refinely evaluate myapp --model gpt-4o`
  runs the app with that model while the LLM judge keeps using the configured
  judge model; `--models a,b,c` records one run per model; `model_name` is a
  column on `evaluation_runs`, so `compare` can show the same config across
  models. Config files never contain a model name.

## Contract notes

- **Configs are plain dicts.** The search space suggests values; your adapter
  maps them onto your runtime. Refinely only guarantees the keys exist in every
  sampled config and that the default config uses the same keys.
- **Weights sum to 1.0**; a missing metric is treated as 0.0, so every metric
  named in the weights must appear in the metrics list.
- **Scores clamp to 0.0–1.0** by construction (weighted means of per-case
  scores); per-case errors and metric failures score 0.0 without aborting the
  run, and the error message is persisted in `case_results.error`.
- **Settings**: refinely's `Settings` (`REFINELY_*` env prefix, `.env` at the
  repo root, `OPENAI_API_KEY` fallback) is only used by refinely's own CLI and
  defaults. A driver passes its own client and settings — the framework never
  reads your configuration.
- **Structured output**: `AsyncOpenAIClient.chat_structured` takes a pydantic
  response model, forces JSON via the prompt, strips fences, and retries once on
  parse failure — no `response_format` dependency.

## DSPy compile (optional)

`refinely` includes an optional DSPy integration for optimizing LLM *behavior* (prompts, demonstrations, reasoning patterns) separately from Optuna's config search.

### Declaring a DSPy program (Style 1 — registry app)

Add a `dspy_factory` field to your `AppRegistration` that returns a `DspyProgramSpec`:

```python
import dspy
from refinely.dspy.spec import DspyProgramSpec
from refinely.eval.datasets import EvalCase


def _my_dspy_factory(settings) -> DspyProgramSpec:
    def build():
        return dspy.Predict("question, context -> answer")

    def prepare_example(case: EvalCase):
        from refinely.dspy.bridge import CASE_ATTR
        ex = dspy.Example(
            question=case.input["question"],
            context=case.input.get("context", ""),
        ).with_inputs("question", "context")
        ex[CASE_ATTR] = case   # embed the original case for metric scoring
        return ex

    def prediction_to_output(pred) -> dict:
        return {"answer": pred.answer}

    return DspyProgramSpec(
        build=build,
        prepare_example=prepare_example,
        prediction_to_output=prediction_to_output,
    )


register_app(AppRegistration(
    name="myapp",
    build_adapter=_build_adapter,
    metrics_factory=_metrics_factory,
    search_space=sample_myapp_config,
    default_config=DEFAULT_CONFIG,
    weights=WEIGHTS,
    dspy_factory=_my_dspy_factory,   # optional; omit if you don't want compile
))
```

Then compile and use the artifact:

```bash
uv sync --group dspy
uv run refinely compile myapp --max-examples 30 --output-dir ./artifacts
uv run refinely evaluate myapp --program ./artifacts/optimized_program.json
```

Compile lineage is stored in `dspy_compiles` (separate from `evaluation_runs`) and queryable:

```python
db = LineageDB(lineage_path)
db.init_schema()
best = db.best_compile("myapp")   # highest compiled_score + parsed config
```

### Using the compile harness programmatically (Style 2 — library driver)

```python
from refinely.dspy.compile import compile_program
from refinely.dspy.spec import DspyProgramSpec
from refinely.eval.datasets import load_dataset, dataset_version

dataset = load_dataset("path/to/myapp_v1.json")
result = compile_program(
    app_name="myapp",
    dataset=dataset,
    dataset_version=dataset_version("path/to/myapp_v1.json"),
    client=client,
    settings=settings,
    metrics=metrics,          # explicit; defaults to registration.metrics_factory
    weights=WEIGHTS,          # explicit; defaults to registration.weights
    max_examples=30,
    output_dir="./artifacts",
)
print(f"baseline {result.baseline_score:.4f} → compiled {result.compiled_score:.4f}")
print(f"artifact: {result.artifact_path}")
```

**Note**: DSPy uses LiteLLM directly (not refinely's `AsyncOpenAIClient`). The `configure_lm` helper wires `dspy.LM` from refinely's `Settings` (same `model_name` and `base_url`), but LiteLLM's API key expectations may differ from your gateway — check that your `REFINELY_OPENAI_API_KEY` value satisfies LiteLLM's key format.
