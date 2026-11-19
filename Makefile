BUILD_DIR  ?= build
BUILD_TYPE ?= Release
VENV       ?= .venv
PYTHON     ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
HOST       ?= 127.0.0.1
PORT       ?= 8000

.DEFAULT_GOAL := help
.PHONY: all build test test-cpp test-py venv serve clean help

all: build

build: ## Build C++ binaries and tests
	@test -f $(BUILD_DIR)/CMakeCache.txt || cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR) -j

test: test-cpp test-py ## Run all C++ and Python tests

test-cpp: build ## Run C++ unit tests
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-py: build ## Run Python unit and integration tests
	$(PYTHON) -m unittest discover -s tests -v

venv: ## Create .venv and install Python dependencies
	@test -x $(VENV)/bin/python || python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt -r scripts/requirements-ingest.txt

serve: build ## Start the FastAPI OEM server
	$(PYTHON) -m uvicorn api.server:app --host $(HOST) --port $(PORT)

clean: ## Remove build directory and Python caches
	rm -rf $(BUILD_DIR)
	find api scripts sdk tests -type d -name __pycache__ -prune -exec rm -rf {} +

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-10s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
