## ADDED Requirements

### Requirement: Retrieval metrics
The system SHALL provide two deterministic retrieval metrics for RAG-style apps: `retrieval_recall` (fraction of expected source snippets that were retrieved) and `citation_accuracy` (precision of cited snippets against expected sources).

#### Scenario: Retrieval recall scoring
- **WHEN** a case's expected output lists source indices and the app output lists retrieved indices
- **THEN** `retrieval_recall` SHALL equal the size of the intersection of the two sets divided by the number of expected source indices, and 0.0 when no expected source is retrieved

#### Scenario: Citation accuracy scoring
- **WHEN** an app output cites snippets
- **THEN** `citation_accuracy` SHALL equal the size of the intersection of cited and expected source indices divided by the number of cited indices
- **WHEN** the app output cites no snippets
- **THEN** `citation_accuracy` SHALL be 0.0

### Requirement: Weighted evaluation for six-metric apps
The system SHALL support evaluation of apps whose weight scheme combines six metrics, with the weights summing to 1.0.

#### Scenario: RAG weight scheme
- **WHEN** the evaluation engine scores a RAG run
- **THEN** the weights SHALL be fuzzy_match 0.20, llm_judge 0.20, retrieval_recall 0.25, citation_accuracy 0.10, latency 0.10, and cost 0.15, summing to 1.0

#### Scenario: Missing metric scores zero
- **WHEN** a case's output cannot be scored by a metric in the scheme
- **THEN** that metric SHALL contribute 0.0 to the case's weighted score, as with existing apps
