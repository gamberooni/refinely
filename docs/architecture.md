# Crucible — Architecture

## 1. High-level system concepts

Crucible is organized as five layers, each with a narrow seam (a Protocol or a registry):

```
CLI / optimize → evaluation → apps → LLM client → provider gateway
                       ↓
                  tracking (SQLite lineage)
```

**Application adapter.** An app is any object with `execute(input: dict, config: dict) -> Result` — duck-typed, no protocol class (the framework calls the method on whatever the app registration's `build_adapter` returned). A `Result` carries the app's `output` (dict or str), `token_usage`, and `latency_seconds`. This duck typing makes every app — from a single structured call to a four-stage pipeline — look identical to the evaluation engine.

**Sync-over-async facade.** The public app API is synchronous; apps internally wrap the async `AsyncOpenAIClient` with `asyncio.run(...)`. The RAG app is the exception: its whole pipeline (expansion → retrieval → rerank → generation) runs inside one `asyncio.run` per case so token usage and latency can be aggregated across its calls.

**Configuration.** A config is a plain dict of parameter values, e.g. `{"temperature": 0.0, "top_k": 3, "system_prompt_variant": "strict"}`. Each app declares its search space, default config, metric set, and weight scheme in its `apps/*.py` module and registers them via `register_app` (`registry.py`). Named configs can be saved as versionable JSON files under `configs/<app>/<name>.json` (managed through the `config` CLI group) and referenced by name with `--config`; when `--config` is omitted, `evaluate` falls back to a per-app default pointer (`configs/<app>/.default`) or the app's registered default config. `optimize` writes the best trial's config to the reserved `configs/<app>/opt-best.json`.

**Command-line interface.** `crucible` is a `click` group living in the `src/crucible/cli/` package. Importing the package runs `discover_apps()` to load registered apps, then wires subcommands across focused modules: `evaluate`, `optimize`, `compile`, `config_cmds` (the `config` group), `readback` (`show` / `compare` / `export`), and `devtools` (`new`, `doctor`, `dataset`). Shared helpers (`_client`, `_load_run_context`, `_resolve_run_id`, ...) live in `cli/context.py`, which acts as a call-time seam so commands resolve helpers through `context.X(...)` at call time and tests can monkeypatch them.

**Dataset.** A versioned JSON file: `{"version": "rag_v1", "corpus": [...], "cases": [{"id", "input", "expected"}]}`. `corpus` is optional (retrieval apps only). Retrieval apps use 0-based corpus indices as snippet identity, so RAG `expected` carries `source_indices`.

**Metrics and weight schemes.** A metric is a `Metric` protocol: `evaluate(case, output) -> MetricResult`. Metrics are pure score functions (deterministic) or LLM judges (`llm_judge`). The framework ships four generic metrics (fuzzy match, LLM judge, latency, cost); app-specific metrics (exact match, retrieval recall, citation accuracy) live in the app modules that register them. Each app bundles its metric set and weight scheme — over its metric names, summing to 1.0 — plus its search space and default config into an `AppRegistration` via `register_app`; the aggregate score is the weighted mean of per-case weighted sums. A failing case or metric scores 0.0 and never aborts the run.

**Lineage.** Every run — baseline evaluation or optimization trial — is written to SQLite: `evaluation_runs` (config, aggregate score, optional `optuna_trial_number`), `metric_results`, and `case_results`. `evaluation_runs` also carries the run's `model_name` (the model axis is orthogonal to config, so the same config can be compared across models) and normalized `tags`; `case_results` includes a nullable `error` column so per-case failures are persisted, not just scored as 0.0. DSPy compile artifacts have their own table `dspy_compiles` (compile_id, optimizer, baseline/compiled scores, artifact path) — kept separate from evaluation runs because compile and Optuna are independent optimization axes. The same SQLite file also stores Optuna's own tables, so trial history and lineage live together. Schema upgrades are additive: `LineageDB._backfill_columns` runs guarded `ALTER TABLE ... ADD COLUMN` statements so pre-existing databases gain new columns without losing rows.

**DSPy compile (optional).** A second, orthogonal optimizer: DSPy optimizes *LLM behavior* (prompts, demos, reasoning patterns via `BootstrapFewShot`) while Optuna optimizes *application config* (temperatures, top_k, strategy flags). Apps that declare a `dspy_factory` on their `AppRegistration` return a `DspyProgramSpec` (three callables: `build` fresh program, `prepare_example` case→dspy.Example, `prediction_to_output` pred→app output dict). The compile harness (`src/crucible/dspy/`) splits the dataset 70/30, runs a baseline evaluation on val, compiles with `BootstrapFewShot` using a metric bridge that scores via the app's registered metrics (same objective as `crucible evaluate`), evaluates the compiled program on val, saves the artifact JSON, and records compile lineage. At evaluate time, `--program <path>` loads the artifact into `build_adapter`; apps fall back to hardcoded prompts when no program path is given. DSPy is an optional dependency (`uv sync --group dspy`); all `import dspy` calls are lazy and raise a clear `EvalError` with install instructions if the group is absent.

**Optimization.** An Optuna study per app (`crucible_{app}`, TPE sampler, `load_if_exists=True` so runs resume). Each trial samples a config, runs a full evaluation, records lineage with the trial number, and returns the aggregate score for maximization. When the study finishes, the best trial's config is written to `configs/<app>/opt-best.json` (overwritten each run) and its path is printed in the CLI output.

**Structured output.** Apps that need JSON responses use `chat_structured` with a pydantic response model; the client forces JSON via schema in the prompt, strips markdown fences, and retries with a repair pass on parse failure.

## 2. Component diagram

```mermaid
flowchart LR
    subgraph CLI["CLI layer"]
        C["crucible (cli/ package)<br/>evaluate / optimize / compile<br/>config / show / compare / export<br/>new / doctor / dataset"]
    end

    subgraph EVAL["Evaluation layer"]
        RUN["EvaluationRunner"]
        MET["Metrics<br/>4 generic impls"]
        REG["App registry (register_app)<br/>metrics / weights / search space / default config / dspy_factory"]
        RT["apps/common.py<br/>keyword / hybrid scorer"]
    end

    subgraph OPT["Optimization layer"]
        OBJ["build_objective"]
        ST["run_study (Optuna TPE)"]
    end

    subgraph DSPY["DSPy layer (optional)"]
        SPEC["DspyProgramSpec<br/>build / prepare_example / prediction_to_output"]
        BRG["metric bridge<br/>make_dspy_metric"]
        COMP["compile_program<br/>BootstrapFewShot"]
    end

    subgraph APPS["App layer"]
        EX["ExtractionApp"]
        QA["QAApp"]
        RA["RAGApp"]
    end

    subgraph LLM["LLM layer"]
        CL["AsyncOpenAIClient<br/>chat_text / chat_structured"]
        GW["OpenAI-compatible<br/>provider / local gateway"]
    end

    subgraph TRACK["Tracking layer"]
        DB["LineageDB (SQLite)<br/>evaluation_runs / metric_results / case_results / dspy_compiles"]
    end

    subgraph DATA["Data"]
        DS["datasets/*_v1.json"]
        CF["configs/<app>/*.json<br/>named configs + opt-best"]
    end

    C --> RUN
    C --> ST
    C --> COMP
    C --> CF
    RUN --> MET
    RUN --> REG
    RUN --> DB
    RUN --> DS
    EX --> CL
    QA --> RT
    QA --> CL
    RA --> RT
    RA --> CL
    CL --> GW
    OBJ --> REG
    OBJ --> RUN
    OBJ --> DB
    ST --> OBJ
    ST --> DB
    COMP --> REG
    COMP --> RUN
    COMP --> BRG
    COMP --> DB
    BRG --> SPEC
    SPEC -. declared by .-> REG
```

## 3. Data flow diagrams

### 3.1 Evaluate flow (baseline run)

```mermaid
flowchart TD
    A["config: --config name / inline JSON / default pointer"] --> RUN["EvaluationRunner.run"]
    DS["dataset JSON<br/>(cases + optional corpus)"] --> RUN
    RUN --> AD["app.execute<br/>per case"]
    AD --> R["Result<br/>output / token_usage / latency"]
    R --> M["metrics (per case)<br/>failures score 0.0"]
    M --> AGG["aggregate_scores<br/>weighted mean"]
    AGG --> OUT["EvaluationRunResult<br/>aggregate + per-metric means"]
    OUT --> DB["LineageDB.record_run"]
    DB --> SQL[("SQLite<br/>evaluation_runs / metric_results / case_results")]
```

### 3.2 Optimize flow (trial loop)

```mermaid
flowchart TD
    ST["Optuna Study<br/>TPESampler, maximize"] -->|"trial"| OBJ["objective (build_objective)"]
    OBJ --> SS["registration.search_space(trial)"]
    SS --> RUN["EvaluationRunner.run<br/>+ metrics"]
    RUN -->|aggregate_score| DB["LineageDB.record_run<br/>optuna_trial_number = trial.number"]
    DB --> SQL[("SQLite<br/>lineage + Optuna tables")]
    RUN -->|aggregate_score| ST
    ST -->|best trial| CLI["CLI output"]
    CLI --> BF["configs/<app>/opt-best.json<br/>best config auto-saved"]
```

## 4. Sequence diagrams

### 4.1 Baseline evaluation (`crucible evaluate extraction`)

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as crucible evaluate
    participant SET as Settings (.env)
    participant RUN as EvaluationRunner
    participant APP as ExtractionApp
    participant LLM as AsyncOpenAIClient
    participant GW as LLM provider
    participant MET as Metrics
    participant DB as LineageDB

    U->>CLI: uv run crucible evaluate extraction
    CLI->>SET: load CRUCIBLE_* settings
    CLI->>RUN: EvaluationRunner(registration.metrics_factory(client, settings), app)
    loop each case
        RUN->>APP: execute(input, config)
        APP->>LLM: chat_structured(model, messages, Model)
        LLM->>GW: HTTP completion request
        GW-->>LLM: completion + usage
        LLM-->>APP: parsed response, TokenUsage
        APP-->>RUN: Result(output, tokens, latency)
        RUN->>MET: evaluate(case, result) for each metric
        MET-->>RUN: MetricResult (score, raw)
    end
    RUN->>RUN: aggregate_scores (weighted mean over cases)
    RUN-->>CLI: EvaluationRunResult
    CLI->>DB: record_run(config, scores, weights)
    CLI-->>U: aggregate_score, metric_results, run id
```

### 4.2 Optimization (`crucible optimize rag --trials 3`)

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as crucible optimize
    participant ST as run_study
    participant TR as Optuna Trial
    participant OBJ as objective
    participant RUN as EvaluationRunner
    participant APP as app (duck-typed)
    participant MET as Metrics
    participant DB as LineageDB

    U->>CLI: uv run crucible optimize rag --trials 3
    CLI->>ST: run_study("rag", objective, n_trials)
    ST->>ST: create or resume study "crucible_rag" (sqlite:///lineage.db)
    loop each trial
        ST->>TR: suggest parameters (TPE)
        TR-->>OBJ: trial
        OBJ->>OBJ: registration.search_space(trial)
        OBJ->>RUN: runner.run(dataset, app, config)
        RUN->>APP: execute per case
        APP-->>RUN: Result per case
        RUN->>MET: score each case
        MET-->>RUN: per-metric results
        RUN-->>OBJ: aggregate_score
        OBJ->>DB: record_run(..., optuna_trial_number=trial.number)
        OBJ-->>ST: aggregate_score (objective value)
    end
    ST-->>CLI: best trial + params
    CLI-->>U: best trial number, score, config
```

### 4.3 RAG app pipeline (one case, one event loop)

```mermaid
sequenceDiagram
    participant RUN as EvaluationRunner
    participant EX as RAGApp.execute (sync)
    participant PIP as _execute_async (asyncio.run)
    participant RT as apps/common.py
    participant LLM as AsyncOpenAIClient
    participant GW as LLM provider

    RUN->>EX: execute(input, config)
    EX->>PIP: asyncio.run(_execute_async)
    PIP->>PIP: validate system_prompt_variant (EvalError before any LLM call)
    alt query_expansion = true
        PIP->>LLM: chat_text (rewrite question)
        LLM->>GW: request
        GW-->>LLM: rewritten question
    end
    PIP->>RT: retrieve_snippets_indexed(question, corpus, top_k, strategy)
    RT-->>PIP: [(idx, snippet)] candidates (0-based corpus indices)
    alt rerank = true and len(candidates) > 1
        PIP->>LLM: chat_structured (SnippetScores)
        LLM->>GW: request
        GW-->>LLM: relevance scores
        PIP->>PIP: reorder candidates by score, keep top_k
    end
    PIP->>LLM: chat_structured (RAGAnswer) with "[snippet {idx}]" blocks
    LLM->>GW: request
    GW-->>LLM: answer + cited_snippets (corpus indices)
    PIP->>PIP: aggregate tokens and latency across all calls
    EX-->>RUN: Result(answer, retrieved_indices, cited_indices, usage, latency)
```

### 4.4 DSPy compile (`crucible compile extraction --max-examples 20`)

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as crucible compile
    participant SET as Settings (.env)
    participant COMP as compile_program
    participant REG as registry
    participant LM as dspy.LM (LiteLLM)
    participant GW as LLM provider
    participant RUN as EvaluationRunner (baseline)
    participant BFS as BootstrapFewShot
    participant BRG as metric bridge
    participant DB as LineageDB

    U->>CLI: uv run crucible compile extraction
    CLI->>SET: load CRUCIBLE_* settings
    CLI->>COMP: compile_program(app_name, dataset, client, settings, ...)
    COMP->>REG: get_registration("extraction") → spec via dspy_factory(settings)
    COMP->>LM: configure_lm(settings, temperature) → dspy.configure(lm)
    COMP->>COMP: _split_train_val(dataset, 70/30, max_examples)
    COMP->>RUN: EvaluationRunner.run(val, build_adapter(client, settings), default_config)
    Note over RUN: baseline score on val via app's registered metrics
    COMP->>BFS: BootstrapFewShot(metric=make_dspy_metric(spec, metrics, weights))
    BFS->>LM: run trainset examples through program
    LM->>GW: completion requests
    GW-->>LM: responses
    BFS-->>COMP: compiled program
    loop each val example
        COMP->>BRG: compiled(**example.inputs()) → prediction_result → score_result
        Note over BRG: uses same registered metrics as EvaluationRunner
    end
    COMP->>COMP: compiled.save("optimized_program.json")
    COMP-->>CLI: CompileResult(baseline_score, compiled_score, artifact_path, ...)
    CLI->>DB: record_compile(app_name, ..., baseline_score, compiled_score)
    CLI-->>U: baseline / compiled scores + artifact path + compile_id
```
