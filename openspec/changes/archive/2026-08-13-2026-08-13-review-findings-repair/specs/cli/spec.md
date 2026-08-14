# CLI Specification — Delta

## MODIFIED Requirements

### Requirement: Evaluate with a compiled program

When an evaluation is run with `--program <path>`, the CLI SHALL load the compiled program through the app's `build_adapter(program_path=...)`. If the artifact is missing, corrupt, or `dspy` is not installed, the CLI SHALL fail with a clean error naming the problem and the install command, not a raw traceback.

#### Scenario: Missing or corrupt artifact
- **WHEN** a user runs `refinely evaluate <app> --program <path>` and the file does not exist, is corrupt, or `dspy` is not installed
- **THEN** the CLI SHALL exit with a clean ClickException-style error explaining the cause and (for a missing dependency) the install command

### Requirement: Compile application via CLI

The `compile` command SHALL accept a `--optimizer {bfs,mipro}` flag (default `mipro`), per-optimizer hyperparameter flags, and a `--min-val` flag (default 5) setting the validation-size floor. The CLI SHALL print the optimizer used, the train/val split sizes, the repeat statistics, and the gate verdict (significant or "n.s."), and SHALL NOT claim improvement when the result is n.s.

#### Scenario: Selecting the optimizer
- **WHEN** a user runs `refinely compile <app> --optimizer mipro`
- **THEN** the CLI SHALL compile with MIPROv2 and report the optimizer name
- **WHEN** a user runs `refinely compile <app> --optimizer bfs`
- **THEN** the CLI SHALL compile with BootstrapFewShot (backwards-compatible path)

## ADDED Requirements

### Requirement: Judge model override

The `evaluate`, `optimize`, and `compile` commands SHALL accept a `--judge-model <name>` option that selects the model used by the LLM judge, independent of the app's generator model (see evaluation-engine: judge model decoupling). Without the option, the judge SHALL use `settings.judge_model`. When the resolved judge model equals the generator model, the CLI SHALL warn loudly unless the user explicitly passed `--judge-model`.

#### Scenario: Overriding the judge model
- **WHEN** a user runs `refinely evaluate <app> --judge-model <other-model>`
- **THEN** the judge SHALL use `<other-model>` and the run's lineage SHALL record it

#### Scenario: Judge equals the generator
- **WHEN** an evaluation runs and the resolved judge model equals the app's generator model without an explicit `--judge-model`
- **THEN** the CLI SHALL warn loudly that the judge is scoring with the same model that generated the answers
