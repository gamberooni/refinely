## ADDED Requirements

### Requirement: RAG app in the CLI registry
The system SHALL register the `rag` app in the CLI so its dataset can be evaluated and optimized with the same commands as existing apps.

#### Scenario: Evaluate and optimize rag
- **WHEN** a user runs `refinely evaluate rag` or `refinely optimize rag`
- **THEN** the CLI SHALL resolve the rag dataset and corpus, build the RAG app with the shared client, and run the standard evaluation or optimization flow, recording the run in lineage
