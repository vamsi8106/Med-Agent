.PHONY: install test unit-tests lint-check lint-fix format-fix format-check pre-commit ci-check serve db-upgrade db-downgrade db-revision ingest-guideline

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

format-check:
	uv run ruff format --check .

pre-commit: format-fix lint-fix lint-check unit-tests

# CI must fail on drift, not silently fix it -- no --fix, no format rewriting.
ci-check: format-check lint-check unit-tests

serve:
	uv run uvicorn medagent.app:app --reload

db-upgrade:
	uv run alembic upgrade head

db-downgrade:
	uv run alembic downgrade -1

db-revision:
	uv run alembic revision -m "$(name)"

ingest-guideline:
	uv run python -m medagent.cli ingest-guideline --file "$(file)" --source "$(source)"
