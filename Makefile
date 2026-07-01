.DEFAULT_GOAL := help
PYTHON := .venv/bin/python

.PHONY: help check rag kag eval summary demo-boxer demo-dogs clean-cache

# Show the available project shortcuts.
help:
	@echo "Animal Farm RAG/KAG Makefile commands:"
	@echo "  make check"
	@echo "  make rag QUESTION=\"What happens to Boxer?\""
	@echo "  make kag QUESTION=\"What happens to Boxer?\""
	@echo "  make eval ID=boxer_fate MODE=both"
	@echo "  make eval MODE=both"
	@echo "  make summary"
	@echo "  make demo-boxer"
	@echo "  make demo-dogs"
	@echo "  make clean-cache"

# Check the database connection and pgvector setup.
check:
	$(PYTHON) src/db/check_connection.py

# Run a DB-backed RAG answer. Provide QUESTION="...".
rag:
	@if [ -z "$(QUESTION)" ]; then \
		echo "Please provide a question. Example: make rag QUESTION=\"What happens to Boxer?\""; \
	else \
		$(PYTHON) src/generation/generate_answer_db.py "$(QUESTION)"; \
	fi

# Run a DB-backed KAG answer. Provide QUESTION="...".
kag:
	@if [ -z "$(QUESTION)" ]; then \
		echo "Please provide a question. Example: make kag QUESTION=\"What happens to Boxer?\""; \
	else \
		$(PYTHON) src/generation/generate_kag_answer_db.py "$(QUESTION)"; \
	fi

# Run the RAG/KAG evaluation runner. MODE defaults to both.
MODE ?= both
eval:
	@if [ -n "$(ID)" ]; then \
		$(PYTHON) src/evaluation/run_rag_kag_eval.py --id "$(ID)" --mode "$(MODE)"; \
	else \
		$(PYTHON) src/evaluation/run_rag_kag_eval.py --mode "$(MODE)"; \
	fi

# Summarize the latest saved evaluation results.
summary:
	$(PYTHON) src/evaluation/summarize_eval_results.py

# Run the complete Boxer RAG, KAG, evaluation, and summary demo.
demo-boxer:
	$(PYTHON) src/generation/generate_answer_db.py "What happens to Boxer?"
	$(PYTHON) src/generation/generate_kag_answer_db.py "What happens to Boxer?"
	$(PYTHON) src/evaluation/run_rag_kag_eval.py --id boxer_fate --mode both
	$(PYTHON) src/evaluation/summarize_eval_results.py

# Run the relationship-focused Napoleon's dogs evaluation demo.
demo-dogs:
	$(PYTHON) src/evaluation/run_rag_kag_eval.py --id napoleons_dogs_role --mode both
	$(PYTHON) src/evaluation/summarize_eval_results.py

# Remove Python bytecode files and cache folders.
clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
