# Running MedAgent: from your laptop to AWS

Four ways to run the same code, each one closer to production. Do them in order the first time. Each stage assumes the one-time setup below.

| Stage | App runs | Postgres | LLM | URL |
|---|---|---|---|---|
| [1. Local](#1-local-app-on-your-machine) | on your machine (`uvicorn --reload`) | Docker container | Groq (or a local model) | `http://localhost:8000` |
| [2. Docker](#2-docker-everything-in-containers) | Docker container | Docker container | Groq (or a local model) | `http://localhost:8000` |
| [3. kind](#3-kind-kubernetes-on-your-machine) | 3 pods in a local Kubernetes cluster | pod in the cluster | mock (no key needed) | `http://localhost:18080` |
| [4. AWS](#4-aws-eks--rds) | 2–6 pods on EKS | RDS | self-hosted model in the VPC | `https://<your domain>` |

The API is the same in every stage, so the [curl walkthrough in the README](../README.md#walkthrough-curl-commands) works against any of them; change `BASE` to the stage's URL.

---

## 0. One-time setup

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker with Compose, and Node.js/npm (for building two of the MCP servers). Stage 3 also needs [kind](https://kind.sigs.k8s.io/) and `kubectl`; stage 4 also needs the `aws` CLI.

```bash
uv sync                          # Python dependencies into .venv
cp .env.example .env             # then edit: GROQ_API_KEY, ADMIN_BOOTSTRAP_PASSWORD
./scripts/setup_vendor_mcp.sh    # fetches healthcare-mcp and med-research-mcp-suite at pinned commits into vendor/
```

Check the code before running anything:

```bash
make pre-commit                  # format, lint, mypy, layer rules, unit tests
```

---

## 1. Local: app on your machine

The app runs directly on your machine and reloads when you save a file. Postgres (with pgvector) and the three MCP servers run in Docker. Use this stage for day-to-day development.

**Start**

```bash
make docker-up-deps              # postgres + healthcare-mcp + research-mcp + medical-mcp-bridge
```

In `.env`, uncomment the host-port block, so the app on your machine finds the containers (the defaults are container names, which only work inside Docker):

```bash
MEDICAL_MCP_URL=http://localhost:8081
HEALTHCARE_MCP_URL=http://localhost:3001
RESEARCH_MCP_URL=http://localhost:3002
POSTGRES_DSN=postgresql://medagent:medagent@localhost:5432/medagent
```

```bash
make db-upgrade                  # create / update the tables (Alembic)
make serve                       # http://localhost:8000, reloads on save
```

**Check**

```bash
curl -s http://localhost:8000/health     # {"status":"ok"}
curl -s http://localhost:8000/ready      # ok / degraded / down per dependency (Postgres, MCP servers)
```

Open `http://localhost:8000/docs` for the interactive API docs.

**Optional: Langfuse.** Set `LANGFUSE_ENABLED=true` and the project keys in `.env` (README, *Prompt tracing*) and restart; each request's prompts and answers appear in Langfuse. Synthetic data only with a hosted Langfuse.

**Add a guideline (optional)**

```bash
make ingest-guideline file=docs/demo/synthetic-t2dm-guideline.txt source=synthetic-t2dm
```

Only ingest documents you are licensed to use (see `docs/info.md` §13).

**Tests that need this stage running**

```bash
make integration                 # real embeddings + Postgres + medical-mcp bridge
make eval                        # real Groq + real MCP servers (needs GROQ_API_KEY)
make rag-eval                    # retrieval quality over guidelines_pdf/ (starts its own Postgres)
```

**Stop**

```bash
# Ctrl+C stops make serve
make docker-down                 # stops the containers; data stays in the Docker volume
```

If port 5432 is taken by a Postgres already installed on your machine, start with `POSTGRES_HOST_PORT=5433 make docker-up-deps` and use port 5433 in `POSTGRES_DSN`.

---

## 2. Docker: everything in containers

The app runs in a container too, next to Prometheus, Grafana and Tempo. This is how a single production server runs, so use it to check the image and the monitoring.

**Start**

Comment the host-port block in `.env` out again: inside Docker the services find each other by container name.

```bash
make docker-up                   # builds and starts all 8 services
docker compose ps                # wait until every service is "healthy" (the app takes ~1-2 min: it loads the embedding model)
```

The app container runs `alembic upgrade head` itself on startup.

**Check**

```bash
curl -s http://localhost:8000/ready
make docker-logs                 # follow the logs of every service
```

- Grafana: `http://localhost:3000` (user `admin`, password `GRAFANA_ADMIN_PASSWORD`). The **MedAgent Overview** dashboard shows requests, LLM calls, MCP calls and guardrails. Traces: *Explore* → *Tempo*.
- Prometheus: `http://localhost:9090`.

- Langfuse (optional): the same `.env` settings as stage 1 are passed to the container; prompts appear in your Langfuse project.
**Add a guideline (optional)**

```bash
docker compose exec medagent python -m medagent.cli ingest-guideline \
  --file docs/demo/synthetic-t2dm-guideline.txt --source synthetic-t2dm
```

**Stop**

```bash
make docker-down
```

**Production on one server** uses the same files plus an overlay that adds Caddy (HTTPS, the only public ports), required secrets and production settings:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Read [deployment.md](deployment.md) §1–9 first: secrets, where the model runs, disk encryption and backups, retention.

---

## 3. kind: Kubernetes on your machine

The Kubernetes manifests that AWS uses, on a cluster that runs inside one Docker container. Use this stage to check that several copies of the app work together before you deploy to AWS: they share approval drafts, abuse limits and guidelines through Postgres, and only one copy applies the migrations.

What the local overlay (`deploy/k8s/overlays/local/`) changes relative to AWS:

| AWS | kind |
|---|---|
| RDS | a pgvector Postgres pod in the cluster |
| ALB + HTTPS | NodePort, published on `localhost:18080` |
| images from ECR | images built locally and loaded into the cluster |
| self-hosted model | `LLM_PROVIDER=mock`: no key, no quota, fixed answers |
| 2–6 replicas | 3 replicas |

**Start**

```bash
cp deploy/k8s/overlays/local/secrets.env.example  deploy/k8s/overlays/local/secrets.env
cp deploy/k8s/overlays/local/postgres.env.example deploy/k8s/overlays/local/postgres.env
# edit both (gitignored): the Postgres password must be the same in the two files

make k8s-local-up
```

`make k8s-local-up`:
1. builds the 4 images with `docker compose build`;
2. creates the kind cluster `medagent`, unless it exists;
3. loads the images into the cluster (`kind load docker-image`, no registry);
4. applies the manifests (`kubectl apply -k deploy/k8s/overlays/local`);
5. waits for the app rollout (up to 10 minutes).

Each app pod's `migrate` init container runs `alembic upgrade head` first. If the pods start before Postgres is ready, they restart a few times and then come up; that is expected.

**Check**

```bash
kubectl -n medagent-local get pods                       # 3 medagent, 3 MCP servers, postgres: all Running
curl -s http://localhost:18080/ready
kubectl -n medagent-local logs deploy/medagent -c migrate # migration output
kubectl -n medagent-local logs -f deploy/medagent         # app logs (one pod)
```

Check that the copies of the app work together:

```bash
# Kill one app pod: requests keep working (3 replicas, PodDisruptionBudget), and a new pod replaces it.
kubectl -n medagent-local delete pod "$(kubectl -n medagent-local get pods -l app.kubernetes.io/name=medagent -o name | head -1 | cut -d/ -f2)"
curl -s http://localhost:18080/ready
```

Run the README walkthrough with `BASE=http://localhost:18080`. Reports come from the mock LLM, so they test the plumbing, not the answers.

**Add a guideline (optional)**: run the CLI on your machine against the cluster's Postgres through a port-forward:

```bash
kubectl -n medagent-local port-forward svc/postgres 15432:5432 &
POSTGRES_DSN="postgresql://medagent:<password from postgres.env>@localhost:15432/medagent" \
  uv run python -m medagent.cli ingest-guideline --file docs/demo/synthetic-t2dm-guideline.txt --source synthetic-t2dm
kill %1
```

Every app pod finds it, because the guidelines are in Postgres.

**Not tested on kind**
- **NetworkPolicies.** kind's default network plugin ignores them. To test them, create the cluster with the default network plugin disabled and install Calico or Cilium.
- **The AWS-only parts:** ALB, HTTPS, RDS, ECR and External Secrets.
- **Answer quality.** The LLM is a mock; `make eval` covers the real model.

**Stop**

```bash
make k8s-local-down              # deletes the whole cluster, including its Postgres data
```

After changing code, run `make k8s-local-up` again. It rebuilds and reloads the images, but pods only restart if the manifests changed, so restart them:

```bash
kubectl -n medagent-local rollout restart deployment/medagent
```

---

## 4. AWS: EKS + RDS

The same manifests with the AWS overlay (`deploy/k8s/overlays/aws/`). [deployment.md](deployment.md) §10 is the full runbook; this is the order of the commands.

**What must exist first** (the overlay does not create them)
- an **EKS** cluster with the **AWS Load Balancer Controller**, and network policies turned on in the VPC CNI;
- **RDS for PostgreSQL** 15.2+ or 16 in the same VPC, with encryption at rest and automated backups;
- an **ACM** certificate for your domain;
- four **ECR** repositories: `medagent`, `medical-mcp-bridge`, `healthcare-mcp`, `research-mcp`;
- a **model** you host inside the VPC (vLLM or Ollama on a GPU node). Production refuses Groq or any public endpoint for patient data unless `ALLOW_EXTERNAL_LLM_WITH_PHI=true`, which states you have a data agreement with that provider.

**1. Enable pgvector on RDS.** The migration runs `CREATE EXTENSION IF NOT EXISTS vector`, which needs `rds_superuser`. If the app's database user does not have it, a DBA runs this once:

```sql
CREATE EXTENSION vector;
```

**2. Build and push the images**

```bash
ACCOUNT=<account id>; REGION=<region>; VERSION=<version, e.g. the git sha>
REGISTRY=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker compose build medagent medical-mcp-bridge healthcare-mcp research-mcp
for image in medagent medical-mcp-bridge healthcare-mcp research-mcp; do
  docker tag "medagent-$image:latest" "$REGISTRY/$image:$VERSION"
  docker push "$REGISTRY/$image:$VERSION"
done
```

**3. Fill in every `<...>` placeholder** in `deploy/k8s/overlays/aws/`:
- `kustomization.yaml`: image URIs and tag, `FORWARDED_ALLOW_IPS` (the VPC CIDR), `LLM_BASE_URL` and `LLM_MODEL`, `OTEL_EXPORTER_OTLP_ENDPOINT`;
- `ingress.yaml`: the ACM certificate ARN and the host name (and the WAF ACL, if you use one).

```bash
grep -rn '<' deploy/k8s/overlays/aws/      # nothing left means you are done
```

**4. Create the namespace and the Secret.** The keys are in `deploy/k8s/base/secret.example.yaml`. `POSTGRES_DSN` points at RDS and ends with `?sslmode=require`.

```bash
kubectl create namespace medagent
kubectl -n medagent create secret generic medagent-secrets --from-env-file=secrets.env
rm secrets.env                   # never commit it
```

Or sync the Secret from AWS Secrets Manager with `overlays/aws/external-secret.example.yaml`.

**5. Deploy**

```bash
kubectl apply -k deploy/k8s/overlays/aws
kubectl -n medagent rollout status deployment/medagent
kubectl -n medagent get pods,ingress      # the ingress shows the ALB's address
```

**6. Point DNS** for your domain at the ALB address, then check:

```bash
curl -s https://<your domain>/health
curl -s https://<your domain>/ready      # the ALB health check uses this too
curl -s -o /dev/null -w '%{http_code}\n' https://<your domain>/metrics   # 404: not public
```

**7. First login.** Log in as the bootstrap admin, create named admin and doctor accounts, then disable the bootstrap account ([deployment.md](deployment.md) §8).

**Add a guideline**: run the CLI from a machine that can reach RDS (for example a bastion in the VPC), with `POSTGRES_DSN` pointing at RDS:

```bash
POSTGRES_DSN="postgresql://...rds.amazonaws.com:5432/medagent?sslmode=require" \
  uv run python -m medagent.cli ingest-guideline --file <guideline.pdf> --source <name>
```

**Updating to a new version**: build and push with a new `VERSION`, change the tag in `kustomization.yaml`, and run `kubectl apply -k deploy/k8s/overlays/aws` again. Pods are replaced one at a time (`maxUnavailable: 0`), each new pod's init container applies any new migrations first, and a doctor whose WebSocket is cut reconnects with `?resume=1` and gets their draft back.

**Before real patients**, also go through [deployment.md](deployment.md): §4 the model, §5 encryption and backups, §7 retention, §11 outbound traffic, §13 known limits.

---

## Quick reference

| Task | Local | Docker | kind | AWS |
|---|---|---|---|---|
| start | `make docker-up-deps` + `make serve` | `make docker-up` | `make k8s-local-up` | `kubectl apply -k deploy/k8s/overlays/aws` |
| stop | Ctrl+C, `make docker-down` | `make docker-down` | `make k8s-local-down` | `kubectl delete -k deploy/k8s/overlays/aws` (also deletes the namespace and its Secret; RDS is untouched) |
| migrations | `make db-upgrade` | on app startup | `migrate` init container | `migrate` init container |
| logs | terminal | `make docker-logs` | `kubectl -n medagent-local logs -f deploy/medagent` | `kubectl -n medagent logs -f deploy/medagent` |
| ingest a guideline | `make ingest-guideline ...` | `docker compose exec medagent python -m medagent.cli ingest-guideline ...` | CLI via `port-forward svc/postgres` | CLI from inside the VPC |
| URL | `http://localhost:8000` | `http://localhost:8000` | `http://localhost:18080` | `https://<your domain>` |
