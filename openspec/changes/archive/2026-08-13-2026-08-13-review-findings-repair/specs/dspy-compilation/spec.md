# dspy-compilation Specification — Delta

## MODIFIED Requirements

### Requirement: DSPy compile harness

The system SHALL provide a compile pipeline that configures a `dspy.LM` from refinely Settings (model name, OpenAI-compatible `base_url`, API key), splits the app's dataset into train and validation subsets, runs an optimizer whose metric scores predictions through the app's registered metrics, and saves the compiled program to a JSON artifact. The default optimizer SHALL be MIPROv2 with BootstrapFewShot available via a `--optimizer {bfs,mipro}` flag. The pipeline SHALL enforce a validation-size floor (default 5, `--min-val` overridable) and SHALL run the baseline and compiled programs for at least 3 repeats on the validation split, reporting mean±std; when the confidence intervals overlap the result SHALL be recorded as not significant ("n.s.") and the CLI SHALL NOT claim improvement.

#### Scenario: Compiling an app program
- **WHEN** the compile pipeline runs for an app with a `dspy_factory`, a bounded training set, and optimizer settings
- **THEN** the pipeline SHALL produce a compiled program saved via `dspy`'s save mechanism, and SHALL return the artifact path plus the baseline and compiled aggregate scores on the validation subset (with repeat statistics and the n.s. flag)

#### Scenario: Dataset too small for a valid comparison
- **WHEN** the validation split would be smaller than the configured floor
- **THEN** the pipeline SHALL fail with a clear error stating the required minimum number of cases

#### Scenario: Metric bridge uses the app's registered metrics
- **WHEN** the optimizer metric evaluates a prediction for a dataset case
- **THEN** the prediction SHALL be mapped to the app's output shape, scored by the app's registered metric set, and reduced to the weighted aggregate score, with the metric returning the score plus feedback (judge rationale and failing sub-metrics) to the optimizer

#### Scenario: DSPy is not installed
- **WHEN** the compile pipeline is invoked without the `dspy` dependency installed
- **THEN** the system SHALL fail with a clear error naming the install command, without importing or touching any dspy code paths in core modules

#### Scenario: Compile objective measures real usage
- **WHEN** the optimizer metric scores a prediction
- **THEN** the metric SHALL use the token usage captured from the dspy LM's forward pass so cost and latency are scored with real numbers; when usage is unavailable in the current dspy version, cost SHALL drop out of the compile objective and SHALL be marked `n/a` in the comparison (never scored as a constant)

#### Scenario: RAG compile objective excludes retrieval
- **WHEN** the compile pipeline trains the RAG program
- **THEN** `retrieval_recall` SHALL be excluded from the training objective (the program does not retrieve); the final `evaluate --program` comparison SHALL retain `retrieval_recall` with real indices from the app-level retrieval

### Requirement: Compiled program consumption

The system SHALL allow an app to consume a compiled program artifact via an optional `program_path` argument on `build_adapter`; when present, the app SHALL load the compiled program and use it in place of its hardcoded prompts, and when absent SHALL use its default behavior. Compiled runs SHALL report real token usage captured from the dspy LM; when usage is unavailable, the cost metric SHALL be excluded from the compiled run's aggregate and marked `n/a` in lineage and read-back.

#### Scenario: Evaluate uses a compiled artifact
- **WHEN** an evaluation is run with `--program <path>` for an app whose `build_adapter` accepts `program_path`
- **THEN** the app SHALL load the program from that path and produce results through the compiled program, reporting real token usage and real latency

#### Scenario: Artifact path absent
- **WHEN** an evaluation is run without `--program`
- **THEN** the app SHALL use its default prompts and behavior, unchanged from before this capability existed

#### Scenario: Compiled artifact fails to load
- **WHEN** an evaluation is run with `--program <path>` and the artifact is missing, corrupt, or `dspy` is not installed
- **THEN** the CLI SHALL fail with a clean error naming the problem and the install command, not a raw traceback
