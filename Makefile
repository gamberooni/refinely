.PHONY: install install-dspy test evaluate optimize compile clean

TRIALS ?= 10
PROGRAM ?=

install: ## Install dependencies (including dev group)
	uv sync --group dev

install-dspy: ## Install optional DSPy dependency group
	uv sync --group dspy

test: ## Run the full test suite (no live API calls)
	uv run pytest tests/ -q

evaluate: ## Evaluate an app: make evaluate APP=extraction [PROGRAM=path/to/optimized_program.json]
	uv run crucible evaluate $(APP) $(if $(PROGRAM),--program $(PROGRAM),)

optimize: ## Optimize an app: make optimize APP=qa [TRIALS=15]
	uv run crucible optimize $(APP) --trials $(TRIALS)

compile: ## Compile a DSPy program: make compile APP=extraction [MAX_EXAMPLES=20] [MAX_ROUNDS=1] (MAX_EXAMPLES must be >= 2)
	uv run crucible compile $(APP) \
		$(if $(MAX_EXAMPLES),--max-examples $(MAX_EXAMPLES),) \
		$(if $(MAX_ROUNDS),--max-rounds $(MAX_ROUNDS),) \
		$(if $(MAX_BOOTSTRAPPED_DEMOS),--max-bootstrapped-demos $(MAX_BOOTSTRAPPED_DEMOS),) \
		$(if $(MAX_LABELED_DEMOS),--max-labeled-demos $(MAX_LABELED_DEMOS),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

clean: ## Remove caches and the local lineage database
	rm -rf .pytest_cache .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f lineage.db
