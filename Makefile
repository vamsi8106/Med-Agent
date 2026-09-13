.PHONY: install test unit-tests lint-check lint-fix format-fix pre-commit serve

install:
	uv sync

test:
	uv run pytest

unit-tests:
	uv run pytest tests/unit/

lint-check:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format-fix:
	uv run ruff format .

pre-commit: format-fix lint-fix lint-check unit-tests

serve:
	uv run uvicorn medagent.app:app --reload
