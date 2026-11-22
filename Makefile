.PHONY: purge-expired deps-audit mcp-audit import-check install test unit-tests eval integration rag-eval agent-eval lint-check lint-fix format-fix format-check type-check pre-commit ci-check serve docker-up-deps docker-up docker-down docker-logs db-upgrade db-downgrade db-revision ingest-guideline k8s-local-up k8s-local-down tf-bootstrap tf-init tf-plan tf-apply tf-destroy tf-check aws-overlay aws-overlay-check tf-down aws-wait-alb aws-leftovers

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

# AWS infrastructure (deploy/terraform): VPC, EKS, RDS, ECR, ACM, Secrets
# Manager, the ALB controller and External Secrets. Needs AWS credentials and
# Terraform >= 1.11. TF_ENV picks deploy/terraform/$(TF_ENV).tfvars (copy the
# .example) and the state key. Runbook: deploy/terraform/README.md.
TF := terraform -chdir=deploy/terraform
TF_ENV ?= demo
AWS_REGION ?= us-east-1
TRIVY_IMAGE := aquasec/trivy:0.76.1

tf-bootstrap:  # once per account: the state bucket
	terraform -chdir=deploy/terraform/bootstrap init
	terraform -chdir=deploy/terraform/bootstrap apply -var region=$(AWS_REGION)

tf-init:
	@test -n "$(TF_STATE_BUCKET)" || { echo "set TF_STATE_BUCKET (make tf-bootstrap prints it)"; exit 1; }
	$(TF) init -reconfigure \
	  -backend-config="bucket=$(TF_STATE_BUCKET)" -backend-config="key=medagent/$(TF_ENV).tfstate" \
	  -backend-config="region=$(AWS_REGION)" -backend-config="encrypt=true" -backend-config="use_lockfile=true"

tf-plan:
	$(TF) plan -var-file=$(TF_ENV).tfvars -out=$(TF_ENV).tfplan

tf-apply:
	$(TF) apply $(TF_ENV).tfplan

tf-destroy:
	$(TF) destroy -var-file=$(TF_ENV).tfvars

# The whole stage-5 teardown, in the order that leaves nothing billing: the
# app first (so the controller deletes the ALB), wait until the ALB is gone,
# destroy the infrastructure, then list anything left.
tf-down:
	-kubectl delete -k deploy/k8s/overlays/aws-generated --ignore-not-found
	scripts/aws_teardown.sh wait-alb medagent-$(TF_ENV)
	$(TF) destroy -var-file=$(TF_ENV).tfvars
	@# EKS can write its log group again while the cluster shuts down.
	-aws logs delete-log-group --log-group-name /aws/eks/medagent-$(TF_ENV)/cluster 2>/dev/null
	scripts/aws_teardown.sh leftovers medagent-$(TF_ENV) $(domain)

# Teardown helpers for either AWS stage (cluster=medagent-manual or medagent-<env>).
aws-wait-alb:
	@test -n "$(cluster)" || { echo "usage: make aws-wait-alb cluster=<name>"; exit 1; }
	scripts/aws_teardown.sh wait-alb $(cluster)

aws-leftovers:
	@test -n "$(cluster)" || { echo "usage: make aws-leftovers cluster=<name> [domain=<host>]"; exit 1; }
	scripts/aws_teardown.sh leftovers $(cluster) $(domain)

# Offline (no AWS): formatting, validation, and a security scan of the config.
tf-check:
	terraform -chdir=deploy/terraform fmt -check -recursive
	terraform -chdir=deploy/terraform/bootstrap init -backend=false -input=false >/dev/null
	terraform -chdir=deploy/terraform/bootstrap validate
	$(TF) init -backend=false -input=false >/dev/null
	$(TF) validate
	docker run --rm -v "$(CURDIR)/deploy/terraform:/src:ro" $(TRIVY_IMAGE) config --exit-code 1 --severity HIGH,CRITICAL --skip-dirs "**/.terraform" /src

# overlays/aws-generated from terraform output; version = the image tag pushed.
aws-overlay:
	@test -n "$(version)" || { echo "usage: make aws-overlay version=<image tag>"; exit 1; }
	scripts/render_aws_overlay.sh $(version)

# The renderer against fixed outputs (demo with Groq, production with vLLM):
# every placeholder filled, and the manifests build.
aws-overlay-check:
	@set -e; dir=deploy/k8s/overlays/aws-generated-check; for case in demo production-gpu; do \
	  TF_OUTPUTS=deploy/terraform/testdata/outputs-$$case.json OUT_DIR=$$dir scripts/render_aws_overlay.sh check >/dev/null 2>&1; \
	  kubectl kustomize $$dir > $$dir.yaml; \
	  if grep -n '<' $$dir.yaml; then echo "$$case: placeholder left"; exit 1; fi; \
	  echo "$$case: ok"; rm -rf $$dir $$dir.yaml; \
	done
