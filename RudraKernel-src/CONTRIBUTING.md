# Contributing

## Development Setup
1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -e ".[dev]"`.
3. Run `make test-all` and ensure the suite passes.

## Commit Rules
- Use step-oriented commit messages: `step-NN: short description`.
- Keep changes scoped to one step or one bug-fix.
- Update brain records after step completion.

## Quality Gates
- `ruff check .`
- `ruff format --check .`
- `mypy siege_env`
- `pytest tests/master_suite.py`
