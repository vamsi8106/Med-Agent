.PHONY: install test unit-tests eval lint-check lint-fix format-fix format-check pre-commit ci-check serve docker-up-deps docker-up docker-down docker-logs db-upgrade db-downgrade db-revision ingest-guideline

install:
	uv sync

test:
	uv run pytest

unit-tests:
	uv run pytest tests/unit/

eval:
	uv run pytest tests/eval/ -v

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

# Hybrid local dev: backing services in Docker, medagent itself bare on the
# host (`make serve`) for fast reload. Point .env at the host-port URLs in
# .env.example's "Bare local dev alternative" section when using this.
docker-up-deps:
	docker compose up -d --build postgres chromadb healthcare-mcp research-mcp medical-mcp-bridge

# Full stack in Docker, medagent included -- closest to production topology.
docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

db-upgrade:
	uv run alembic upgrade head

db-downgrade:
	uv run alembic downgrade -1

db-revision:
	uv run alembic revision -m "$(name)"

ingest-guideline:
	uv run python -m medagent.cli ingest-guideline --file "$(file)" --source "$(source)"
