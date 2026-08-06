# dspy-compilation Specification

## Purpose

Optional DSPy integration: per-app `dspy_factory` program declaration, a `BootstrapFewShot` compile harness that scores predictions through the app's registered metrics, and consumption of compiled artifacts via `build_adapter(program_path=...)` at evaluate time. (Adapted from change `dspy-compile-pipeline`.)
## Requirements
### Requirement: Per-app DSPy program declaration
The system SHALL support an optional `dspy_factory` field on app registrations, a callable `dspy_factory(settings)` returning a `DspyProgramSpec` that provides a fresh uncompiled `dspy.Module` builder, an example preparer, and a prediction-to-output mapper. Apps without a `dspy_factory` SHALL remain fully supported for evaluation and Optuna optimization.

#### Scenario: Registered app declares a program
- **WHEN** an app registration includes a `dspy_factory` and `dspy` is importable
- **THEN** the framework SHALL be able to build the app's `dspy.Module`, convert dataset cases into `dspy.Example` inputs via the preparer, and map predictions back into the app's output dict shape via the mapper

#### Scenario: App without a program is unaffected
- **WHEN** an app registration has no `dspy_factory`
- **THEN** evaluation and optimization SHALL behave exactly as before, and the compile command SHALL report that the app does not support compilation

### Requirement: DSPy compile harness
The system SHALL provide a compile pipeline that configures a `dspy.LM` from crucible Settings (model name, OpenAI-compatible `base_url`, API key), splits the app's dataset into train and validation subsets, runs a `BootstrapFewShot` optimizer whose metric scores predictions through the app's registered metrics, and saves the compiled program to a JSON artifact.

#### Scenario: Compiling an app program
- **WHEN** the compile pipeline runs for an app with a `dspy_factory`, a bounded training set, and optimizer settings
- **THEN** the pipeline SHALL produce a compiled program saved via `dspy`'s save mechanism, and SHALL return the artifact path plus the baseline and compiled aggregate scores on the validation subset

#### Scenario: Metric bridge uses the app's registered metrics
- **WHEN** the optimizer metric evaluates a prediction for a dataset case
- **THEN** the prediction SHALL be mapped to the app's output shape, scored by the app's registered metric set, and reduced to the weighted aggregate score

#### Scenario: DSPy is not installed
- **WHEN** the compile pipeline is invoked without the `dspy` dependency installed
- **THEN** the system SHALL fail with a clear error naming the install command, without importing or touching any dspy code paths in core modules

### Requirement: Compiled program consumption
The system SHALL allow an app to consume a compiled program artifact via an optional `program_path` argument on `build_adapter`; when present, the app SHALL load the compiled program and use it in place of its hardcoded prompts, and when absent SHALL use its default behavior.

#### Scenario: Evaluate uses a compiled artifact
- **WHEN** an evaluation is run with `--program <path>` for an app whose `build_adapter` accepts `program_path`
- **THEN** the app SHALL load the program from that path and produce results through the compiled program

#### Scenario: Artifact path absent
- **WHEN** an evaluation is run without `--program`
- **THEN** the app SHALL use its default prompts and behavior, unchanged from before this capability existed
