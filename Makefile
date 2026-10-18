.PHONY: purge-expired deps-audit import-check install test unit-tests eval lint-check lint-fix format-fix format-check type-check pre-commit ci-check serve docker-up-deps docker-up docker-down docker-logs db-upgrade db-downgrade db-revision ingest-guideline

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

type-check:
	uv run mypy

import-check:
	uv run lint-imports

# Known-vulnerability scan of the locked runtime dependencies. torch is skipped
# by pip-audit (the CPU wheel is not on PyPI). The ignored chromadb advisories
# have no fixed release yet and concern the Chroma *server* (we use the package
# as an HTTP client); the server container is reachable only on the compose
# network / localhost. Remove each ignore as soon as a fix ships.
CHROMADB_UNFIXED := PYSEC-2026-311 PYSEC-2026-3813 PYSEC-2026-3814 PYSEC-2026-3815
deps-audit:
	uv export --frozen --no-dev --no-hashes --no-emit-project --quiet \
	  | uv run pip-audit --disable-pip --no-deps --progress-spinner off -r /dev/stdin \
	    $(foreach id,$(CHROMADB_UNFIXED),--ignore-vuln $(id))

pre-commit: format-fix lint-fix lint-check type-check import-check unit-tests

# CI must fail on drift, not silently fix it -- no --fix, no format rewriting.
ci-check: format-check lint-check type-check import-check unit-tests

serve:
	uv run uvicorn medagent.app:app --reload --no-access-log

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

# Retention: dry run (counts only). Deleting needs RETENTION_DAYS /
# AUDIT_RETENTION_DAYS set and `uv run python -m medagent.cli purge-expired --execute`.
purge-expired:
	uv run python -m medagent.cli purge-expired

ingest-guideline:
	uv run python -m medagent.cli ingest-guideline --file "$(file)" --source "$(source)"
