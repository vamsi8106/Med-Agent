.PHONY: purge-expired deps-audit mcp-audit import-check install test unit-tests eval integration rag-eval agent-eval lint-check lint-fix format-fix format-check type-check pre-commit ci-check serve docker-up-deps docker-up docker-down docker-logs db-upgrade db-downgrade db-revision ingest-guideline k8s-local-up k8s-local-down

install:
	uv sync

test:
	uv run pytest

unit-tests:
	uv run pytest tests/unit/

eval:
	uv run pytest tests/eval/ -v

# Real sentence-transformers + throwaway Postgres + the medical-mcp bridge
# (`make docker-up-deps`; from the host set MEDICAL_MCP_URL=http://localhost:8081);
# each test skips if its dependency is unavailable.
integration:
	uv run pytest tests/integration/ -v

# Retrieval quality (recall@k, MRR) over the local guideline PDFs, for three
# chunking configurations side by side. Offline: no LLM, no MCP. Run it before
# and after changing chunking, the embedding model or retrieval.
rag-eval:
	HF_HUB_OFFLINE=1 uv run python -m tests.eval.retrieval_eval

# Agent eval: every case end to end (real LLM + MCP + pgvector), rates and a
# saved report in eval-results/. args="--category safety", "--case rn-02 --repeats 5 --judge".
agent-eval:
	HF_HUB_OFFLINE=1 uv run python -m tests.eval.agent_eval $(args)

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

# npm audit over the three MCP servers' committed lockfiles (docker/*/), the
# exact dependencies their images install with `npm ci --omit=dev`. Fails on
# high or critical. Vendored servers are pinned in scripts/setup_vendor_mcp.sh.
MCP_SERVERS := docker/healthcare-mcp docker/med-research-mcp-suite docker/medical-mcp-bridge
mcp-audit:
	@for dir in $(MCP_SERVERS); do \
	  echo "== $$dir"; \
	  (cd $$dir && npm audit --omit=dev --audit-level=high) || exit 1; \
	done

# Known-vulnerability scan of the locked runtime dependencies. torch is skipped
# by pip-audit (the CPU wheel is not on PyPI). No advisories are ignored.
deps-audit:
	uv export --frozen --no-dev --no-hashes --no-emit-project --quiet \
	  | uv run pip-audit --disable-pip --no-deps --progress-spinner off -r /dev/stdin

pre-commit: format-fix lint-fix lint-check type-check import-check unit-tests

# CI must fail on drift, not silently fix it -- no --fix, no format rewriting.
ci-check: format-check lint-check type-check import-check unit-tests

serve:
	uv run uvicorn medagent.app:app --reload --no-access-log

# Hybrid local dev: backing services in Docker, medagent itself bare on the
# host (`make serve`) for fast reload. Point .env at the host-port URLs in
# .env.example's "Bare local dev alternative" section when using this.
docker-up-deps:
	docker compose up -d --build postgres healthcare-mcp research-mcp medical-mcp-bridge

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

# Multi-replica test on a local kind cluster (deploy/k8s/overlays/local):
# 3 app replicas sharing drafts and limits through Postgres, the mock LLM.
# Needs kind + kubectl, `./scripts/setup_vendor_mcp.sh`, and secrets.env /
# postgres.env copied from their .example files. App: http://localhost:18080
K8S_LOCAL := deploy/k8s/overlays/local
k8s-local-up:
	docker compose build medagent medical-mcp-bridge healthcare-mcp research-mcp
	kind get clusters | grep -qx medagent || kind create cluster --name medagent --config $(K8S_LOCAL)/kind-config.yaml
	for image in medagent medical-mcp-bridge healthcare-mcp research-mcp; do \
	  kind load docker-image --name medagent medagent-$$image:latest; done
	kubectl apply -k $(K8S_LOCAL)
	kubectl -n medagent-local rollout status deployment/medagent --timeout=600s

k8s-local-down:
	kind delete cluster --name medagent
