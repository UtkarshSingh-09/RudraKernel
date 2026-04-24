PYTHON ?= python3
PYTEST ?= $(if $(wildcard .venv/bin/pytest),.venv/bin/pytest,pytest)
RUFF ?= $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)
MYPY ?= $(if $(wildcard .venv/bin/mypy),.venv/bin/mypy,mypy)
STEP ?= 00
TITLE ?= Bootstrap
OWNER ?= Utkarsh
REVIEWER ?= Utkarsh
GATE_TEST ?= pending
MASTER_SUITE ?= pending
COVERAGE ?= 0.0

.PHONY: test-all test-master test-step brain-update lint format

test-all:
	$(PYTEST) tests/master_suite.py -q

test-master:
	$(PYTEST) tests/master_suite.py -q

test-step:
	@if [ -z "$(STEP)" ]; then echo "Usage: make test-step STEP=00"; exit 1; fi
	$(PYTEST) tests/step_tests/step_$(STEP)_*_test.py -q

brain-update:
	$(PYTHON) brain/tools/update_brain.py --step "$(STEP)" --title "$(TITLE)" --owner "$(OWNER)" --reviewer "$(REVIEWER)" --gate-test "$(GATE_TEST)" --master-suite "$(MASTER_SUITE)" --coverage "$(COVERAGE)"

lint:
	$(RUFF) check .
	$(MYPY) siege_env

format:
	$(RUFF) format .
