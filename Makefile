DEV_DIR := .dev

export PYTHONPYCACHEPREFIX := $(DEV_DIR)/cache/python
export UV_CACHE_DIR := $(DEV_DIR)/cache/uv

.PHONY: build test

test:
	uv run python tests/big_projects/run.py

build:
	uv run python -m build --outdir $(DEV_DIR)/dist
