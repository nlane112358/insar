BUILD_DIR  ?= build
BUILD_TYPE ?= Release
VENV       ?= .venv
PYTHON     ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
HOST       ?= 127.0.0.1
PORT       ?= 8000

.DEFAULT_GOAL := help
.PHONY: all build test test-cpp test-py test-unit venv serve compose-up compose-down cli-install clean help

all: build

build: ## Build C++ binaries and tests
	@test -f $(BUILD_DIR)/CMakeCache.txt || cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR) -j

test: test-cpp test-py ## Run all C++ and Python tests

test-cpp: build ## Run C++ unit tests
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-unit: ## Run fast Python unit tests without invoking C++ build
	PYTHONPATH=python $(PYTHON) -m unittest discover -s tests/unit -v

test-py: build ## Run Python unit and integration tests
	PYTHONPATH=python $(PYTHON) -m unittest discover -s tests -v

venv: ## Create .venv and install Python dependencies
	@test -x $(VENV)/bin/python || python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r deploy/requirements.txt -r deploy/requirements-ingest.txt

serve: build ## Start the FastAPI OEM server
	PYTHONPATH=python $(PYTHON) -m uvicorn insar_platform.api.app:app --host $(HOST) --port $(PORT)

compose-up: ## Build and start PostGIS, MinIO, Redis, API, worker, and scheduler
	docker compose -f deploy/docker-compose.yml up -d --build

compose-down: ## Stop the local Compose stack and retain persisted volumes
	docker compose -f deploy/docker-compose.yml down

cli-install: ## Install the insarctl operator command in the active Python environment
	$(PYTHON) -m pip install -e .

clean: ## Remove build directory and Python caches
	rm -rf $(BUILD_DIR)
	find python scripts tests examples -type d -name __pycache__ -prune -exec rm -rf {} +

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-10s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
