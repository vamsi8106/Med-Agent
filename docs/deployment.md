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

## 10. Known limits

- Rate limits and circuit breakers are kept per process. With several app replicas, each replica keeps its own counts.
- Pending WebSocket approval drafts live in memory, so a restart discards them.
- The checks in this runbook cover the application. A real deployment handling patient data also needs an organisational review: access policy, incident response, and the regulations that apply to you, such as HIPAA or GDPR.
