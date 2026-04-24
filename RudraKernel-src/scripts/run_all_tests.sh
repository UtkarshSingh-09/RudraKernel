#!/usr/bin/env bash
set -euo pipefail

pytest tests/master_suite.py --cov=siege_env --cov-report=term-missing