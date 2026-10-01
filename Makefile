VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help install corpus corpus-restore ingest index ask ui serve bench bench-sweep review prescreen calibrate tune-fusion eval-dry test test-all lint doctor config clean

help:
	@grep -E '^[a-zA-Z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

install: ## create the venv and install all dependencies
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt

corpus: ## download the arXiv corpus (respects arXiv rate limits; slow by design)
	$(PY) scripts/fetch_corpus.py --count 40

corpus-restore: ## re-download exactly the manifest's pinned papers (sha256-checked)
	$(PY) scripts/fetch_corpus.py --from-manifest

ingest: ## parse and chunk the corpus into chunks.jsonl
	PYTHONPATH=src $(PY) -m ragpipe.cli ingest

index: ## embed the chunks into the vector store
	PYTHONPATH=src $(PY) -m ragpipe.cli index

ask: ## ask a question: make ask Q="what is self-attention?"
	PYTHONPATH=src $(PY) -m ragpipe.cli ask "$(Q)" --show-context

ui: ## run the Streamlit demo on http://localhost:8501
	PYTHONPATH=src $(VENV)/bin/streamlit run app/streamlit_app.py

serve: ## run the API on http://localhost:8000 (docs at /docs)
	PYTHONPATH=src $(VENV)/bin/uvicorn ragpipe.api.app:app --reload --port 8000

config: ## show the resolved configuration
	PYTHONPATH=src $(PY) -m ragpipe.cli config

doctor: ## check which providers are usable right now
	PYTHONPATH=src $(PY) -m ragpipe.cli doctor

bench: ## compare retrieval configurations (known-item diagnostic)
	$(PY) scripts/bench_retrieval.py --per-family 50

bench-sweep: ## sweep fusion weights
	$(PY) scripts/bench_retrieval.py --sweep --per-family 40 --out eval_results/retrieval_sweep.json

review: ## human-verify the golden set in the browser (the only route to 'verified')
	PYTHONPATH=src $(VENV)/bin/streamlit run scripts/review_golden.py

prescreen: ## LLM pre-screen of golden pairs -- advisory flags only, never verifies
	$(PY) scripts/prescreen_golden.py

calibrate: ## measure the relevance gate on verified pairs (retrieval + rerank, no LLM)
	$(PY) scripts/calibrate_gate.py

tune-fusion: ## fusion-weight sweep on a held-out split (no LLM calls, free)
	$(PY) scripts/tune_fusion.py

eval-dry: ## print the cost estimate for a full eval; calls nothing
	$(PY) scripts/run_eval.py --dry-run

test: ## fast tests only (what the CI gate runs)
	$(PY) -m pytest -q -m "not slow"

test-all: ## include tests that download real model weights
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts

clean:
	rm -rf .pytest_cache .ragpipe_cache **/__pycache__ .coverage
