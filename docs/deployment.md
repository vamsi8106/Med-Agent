# Deploying MedAgent

This runbook takes MedAgent from a development checkout to a production host. The application refuses several unsafe settings by itself (see step 3). Some steps are infrastructure work that code cannot do for you, such as disk encryption. Each step says which kind it is.

## 1. Before you start

- A Linux host with Docker and Docker Compose v2.24 or newer (the prod overlay uses `!reset`).
- A DNS name pointing at the host (for example `medagent.example.org`) and ports 80 and 443 open to the internet. Without a domain you can still try the setup on `https://localhost` with a locally issued certificate.
- A decision on **where the model runs** (step 4). Prompts contain patient data.
- A decision on **how long records are kept** (step 7). This is a legal question in most countries.

## 2. Secrets (required)

Put these in `.env` on the host. Never commit them.

```bash
JWT_SECRET_KEY=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -hex 24)
GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 16)
ADMIN_BOOTSTRAP_USERNAME=admin
ADMIN_BOOTSTRAP_PASSWORD='<12+ characters, stored in your password manager>'
DOMAIN=medagent.example.org
```

The production overlay has no defaults for these. `docker compose` stops with "required variable ... is missing a value" until each one is set. The app also refuses to start with `APP_ENV=production` if `JWT_SECRET_KEY` is the placeholder or shorter than 32 characters, or if the bootstrap password is under 12 characters.

`ADMIN_BOOTSTRAP_*` only creates the first admin, and only when the users table is empty. After the first start, log in as that admin, create named admin and doctor accounts, and disable the bootstrap account if nobody should share it (step 8).

## 3. Start the production stack

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

What this overlay changes compared with development:

| | Development | Production overlay |
|---|---|---|
| Public ports | none (everything on 127.0.0.1) | only Caddy: 80 and 443 |
| TLS | none | Caddy, automatic Let's Encrypt certificate for `DOMAIN` (local CA for `localhost`) |
| App | `APP_ENV=development` | `APP_ENV=production`: strong secrets enforced, `/docs` hidden, HSTS on, external LLM refused without an explicit data agreement |
| Client IPs for rate limits | direct | from Caddy (`--proxy-headers`) |
| `/metrics` | on 127.0.0.1 | not served publicly (Caddy returns 404); Prometheus scrapes it inside the network |
| Grafana | 127.0.0.1:3000 | 127.0.0.1:3000; reach it through an SSH tunnel: `ssh -L 3000:localhost:3000 host` |
| Restart policy | none | `unless-stopped` |

Check it:

```bash
curl -s https://$DOMAIN/health                 # {"status":"ok"}
curl -s https://$DOMAIN/ready                  # ok / degraded / down per dependency
curl -sI https://$DOMAIN/health | grep -i strict-transport-security
```

## 4. The model: keep patient data on your machines (decision required)

Every prompt includes the patient's conditions, medications, allergies and labs. In production the app refuses to start with `LLM_PROVIDER=groq`, or with an `openai_compatible` endpoint on a public host, unless you set `ALLOW_EXTERNAL_LLM_WITH_PHI=true`. That setting states that you have a data-processing agreement with that provider, such as a HIPAA BAA.

**Self-hosted (recommended for real patient data).** The overlay includes an optional Ollama service:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile local-llm up -d
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec ollama ollama pull qwen2.5:7b
```

Then set the following in `.env` and restart `medagent`:

```bash
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://ollama:11434/v1
LLM_MODEL=qwen2.5:7b
```

Any server with an OpenAI-style `/chat/completions` API works the same way (vLLM, LM Studio, a GPU box on your private network). The model **must support tool calling**, because the evidence agent is a ReAct agent. Answer quality depends heavily on the model size. Run `make eval` against your chosen model before relying on it: the evals in this repository were tuned on Groq's `openai/gpt-oss-20b`. Small models (3B and under) run on a CPU but are much weaker. Use a GPU host for 7B and larger.

**External provider.** Only with a signed agreement that covers patient data. Then set `ALLOW_EXTERNAL_LLM_WITH_PHI=true`. Keep `LANGCHAIN_TRACING_V2=false`, because LangSmith would receive full prompts.

## 5. Encryption at rest and backups (infrastructure)

The application cannot encrypt its own disk. Do one of the following:

- Put Docker's data directory, or at least the `postgres-data` volume, on an encrypted disk: LUKS/dm-crypt on your own hardware, or encrypted block storage from your cloud provider (AWS EBS encryption, GCP CMEK, Azure Disk Encryption).
- Run Postgres as a managed service with encryption at rest, and point `POSTGRES_DSN` at it.

ChromaDB holds only guideline text (no patient data). Tempo, Prometheus and logs contain no patient data by design, which is enforced by `tests/unit/infra/test_no_phi_in_telemetry.py`.

**Backups.** They contain everything, so encrypt them and test restores:

```bash
docker compose exec -T postgres pg_dump -U medagent -Fc medagent \
  | gpg --symmetric --cipher-algo AES256 -o medagent-$(date +%F).dump.gpg
# restore: gpg -d file.dump.gpg | docker compose exec -T postgres pg_restore -U medagent -d medagent --clean
```

**Postgres over TLS.** For a database on another host, add `?sslmode=require` (or `verify-full` with a CA) to `POSTGRES_DSN`; asyncpg honours it. Inside the compose network, traffic does not leave the host.

## 6. Upgrades

```bash
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Migrations (`alembic upgrade head`) run automatically when `medagent` starts. Take a backup first. Every migration has a downgrade (`docker compose exec medagent alembic downgrade -1`).

Run `make deps-audit` regularly. CI runs it on every push. When ChromaDB publishes fixes for the advisories listed in the `Makefile`, upgrade the `chromadb` image and package and remove those ignores.

## 7. Retention (decision required)

Nothing is ever deleted unless you configure it:

```bash
RETENTION_DAYS=3650         # whole patient records inactive this long (minimum 365)
AUDIT_RETENTION_DAYS=3650   # audit entries older than this (minimum 365)
```

A record's last activity is the later of its last update and its newest visit. When a record expires it is deleted whole: patient, medications, labs, visits and interactions. Audit entries are kept on their own schedule, so access history can outlive the data. Each purge writes its own audit entry.

```bash
# dry run: prints counts only
docker compose exec medagent python -m medagent.cli purge-expired
# delete
docker compose exec medagent python -m medagent.cli purge-expired --execute
```

Schedule the `--execute` run, for example with a monthly cron job on the host:

```cron
0 3 1 * * cd /srv/medagent && docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T medagent python -m medagent.cli purge-expired --execute
```

## 8. Accounts

All account changes are admin-only and audited:

| Action | Request |
|---|---|
| Create | `POST /admin/users` `{"username", "password", "role": "doctor"\|"admin"}` |
| List | `GET /admin/users` |
| Disable / enable | `POST /admin/users/{username}/disable` / `/enable` (disabling also ends their sessions) |
| Reset a password | `POST /admin/users/{username}/reset-password` `{"new_password"}` |
| End all sessions (lost device) | `POST /admin/users/{username}/revoke-tokens` |
| Change own password (any user) | `POST /auth/change-password` `{"current_password", "new_password"}` |

Passwords need 12 or more characters. Tokens last `JWT_EXPIRE_MINUTES` (60 by default). Disabling a user, resetting or changing a password, and revoking tokens all invalidate existing tokens immediately.

## 9. Monitoring

- `/ready` returns `ok`, `degraded` or `down` per dependency. Point an uptime check at `https://$DOMAIN/ready`. It returns 503 when Postgres is down.
- Grafana (through the SSH tunnel) shows the MedAgent Overview dashboard and Tempo traces.
- Watch `medagent_abuse_limit_hits_total` (login and agent-run limits), `medagent_circuit_breaker_state`, LLM outcomes and the evidence agent's fallback rate.

## 10. Kubernetes and AWS (several replicas)

Steps 1–9 describe one host with Docker Compose. To run several app replicas behind a load balancer, for example on AWS, use the Kustomize manifests in `deploy/k8s/`.

### What makes several replicas safe

| Shared state | With `STATE_BACKEND=postgres` |
|---|---|
| WebSocket approval drafts | LangGraph's Postgres checkpointer (`checkpoint_*` tables). Any replica can resume a draft, and a pod restart keeps it. Drafts still expire after `APPROVAL_DRAFT_TTL_MINUTES`, are deleted with their patient by retention, and are encrypted at rest by the database. |
| Login and agent-run limits | `rate_limit_buckets` table, updated under a row lock. One limit for the whole deployment, not one per pod. |
| Migrations | Each pod's `migrate` initContainer runs `alembic upgrade head`; a Postgres advisory lock makes pods queue, so exactly one applies them. |
| Bootstrap admin | A pod that loses the race to create it carries on. |

Kept **per pod** on purpose:
- **Circuit breakers:** each pod notices an outage on its own, which is the standard design.
- **Outbound MCP rate limits:** `MCP_RATE_LIMIT_PER_SECOND` applies per pod, so N pods make up to N× the calls.

### Layout

```
deploy/k8s/
  base/                 app Deployment (2 replicas, probes, non-root, read-only root fs), Service,
                        HPA (2–6 on CPU), PodDisruptionBudget, MCP servers, ChromaDB StatefulSet,
                        NetworkPolicies, ConfigMap; secret.example.yaml (not applied)
  overlays/local/       kind: in-cluster Postgres, 3 replicas, mock LLM, NodePort → localhost:18080
  overlays/aws/         EKS: ALB Ingress (ACM certificate, HTTPS only, 3600 s idle timeout for
                        WebSockets, /metrics → 404), RDS via POSTGRES_DSN, ECR images
```

### Try it locally (kind)

```bash
./scripts/setup_vendor_mcp.sh                       # once, builds the two vendored MCP servers
cp deploy/k8s/overlays/local/secrets.env.example  deploy/k8s/overlays/local/secrets.env
cp deploy/k8s/overlays/local/postgres.env.example deploy/k8s/overlays/local/postgres.env
# edit both: the Postgres password must match
make k8s-local-up                                   # builds images, creates the cluster, deploys
curl -s http://localhost:18080/ready
make k8s-local-down
```

### Deploy on AWS

The overlay assumes these exist already; it does not create them:
- an **EKS** cluster with the **AWS Load Balancer Controller**;
- **RDS for PostgreSQL** in the same VPC, with encryption at rest and automated backups;
- an **ACM** certificate for your domain;
- **ECR** repositories for the four images.

1. **Build and push the images** (`medagent`, `medical-mcp-bridge`, `healthcare-mcp`, `research-mcp`) to ECR.
2. **Fill in every `<...>` placeholder** in `deploy/k8s/overlays/aws/`:
   - image URIs;
   - `FORWARDED_ALLOW_IPS` (the VPC CIDR, so only the ALB can set the client IP);
   - the model endpoint;
   - the OpenTelemetry collector;
   - the certificate ARN and host name.
3. **Create the Secret**: `kubectl create namespace medagent`, then either
   - `kubectl -n medagent create secret generic medagent-secrets --from-env-file=secrets.env`, using the keys in `base/secret.example.yaml`, with `POSTGRES_DSN` pointing at RDS with `?sslmode=require`; or
   - sync it from AWS Secrets Manager with `overlays/aws/external-secret.example.yaml`.
4. **Deploy**: `kubectl apply -k deploy/k8s/overlays/aws`, then `kubectl -n medagent rollout status deployment/medagent`.
5. **Point DNS at the ALB.** `/ready` is the load balancer's health check.

**Rolling updates.** `maxUnavailable: 0` keeps capacity during a rollout. `preStop` and the 90 s grace period let requests in flight finish. An open WebSocket is cut when its pod stops, and the client reconnects with `?resume=1` to any pod and gets its draft back.

**Model and patient data.** The overlay uses `openai_compatible`, a model you host inside the VPC (step 4). Groq or any public endpoint is refused in production unless `ALLOW_EXTERNAL_LLM_WITH_PHI=true`.

**Observability.** Pods carry Prometheus scrape annotations. Use Amazon Managed Prometheus or your own Prometheus, and send traces through an OpenTelemetry collector (ADOT → X-Ray, or Tempo). The Compose stack's Prometheus, Grafana and Tempo are not part of these manifests.

## 11. Known limits

- Circuit breakers and outbound MCP rate limits are kept per process (by design, see section 10). With `STATE_BACKEND=memory`, abuse limits and drafts are per process too, so use `postgres` with more than one replica.
- With `STATE_BACKEND=memory`, a restart discards pending WebSocket drafts.
- The embedding model is downloaded from Hugging Face when a pod starts, so pods need egress to it (or bake the model into the image).
- The NetworkPolicies (only app pods may reach the MCP servers, ChromaDB and the in-cluster Postgres) take effect only where the network plugin enforces them: on EKS, enable network policies in the VPC CNI (or use Calico or Cilium). The default kind network plugin used by `make k8s-local-up` did not enforce them in testing.
- The checks in this runbook cover the application. A real deployment handling patient data also needs an organisational review: access policy, incident response, and the regulations that apply to you, such as HIPAA or GDPR.
