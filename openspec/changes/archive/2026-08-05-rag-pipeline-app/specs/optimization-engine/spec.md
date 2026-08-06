## ADDED Requirements

### Requirement: Mixed-type search spaces
The system SHALL support per-app Optuna search spaces that combine continuous floats, categoricals, integers, and booleans in a single config sampling function.

#### Scenario: RAG search space
- **WHEN** the optimizer samples a config for the `rag` app
- **THEN** the config SHALL contain `temperature` (float 0.0-1.0), `system_prompt_variant` (categorical strict/verbose), `retrieval_strategy` (categorical keyword/hybrid), `top_k` (int 1-6), `query_expansion` (boolean), and `rerank` (boolean)

#### Scenario: Default config shape
- **WHEN** the CLI runs a baseline evaluation of the `rag` app
- **THEN** the default config SHALL be temperature 0.0, strict variant, hybrid retrieval, top_k 3, query_expansion off, and rerank off
