VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help install corpus test test-all lint doctor config clean

help:
	@grep -E '^[a-zA-Z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

install: ## create the venv and install all dependencies
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt

corpus: ## download the arXiv corpus (respects arXiv rate limits; slow by design)
	$(PY) scripts/fetch_corpus.py --count 40

config: ## show the resolved configuration
	PYTHONPATH=src $(PY) -m ragpipe.cli config

doctor: ## check which providers are usable right now
	PYTHONPATH=src $(PY) -m ragpipe.cli doctor

test: ## fast tests only (what the CI gate runs)
	$(PY) -m pytest -q -m "not slow"

test-all: ## include tests that download real model weights
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts

clean:
	rm -rf .pytest_cache .ragpipe_cache **/__pycache__ .coverage
