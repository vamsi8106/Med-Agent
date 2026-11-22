# Running MedAgent: from your laptop to AWS

Five ways to run the same code, each one closer to production. Do them in order the first time: each stage adds one new idea to the one before. Each assumes the one-time setup below.

| Stage | App runs | Postgres | LLM | URL |
|---|---|---|---|---|
| [1. Local](#1-local-app-on-your-machine) | on your machine (`uvicorn --reload`) | Docker container | Groq (or a local model) | `http://localhost:8000` |
| [2. Docker](#2-docker-everything-in-containers) | Docker container | Docker container | Groq (or a local model) | `http://localhost:8000` |
| [3. kind](#3-kind-kubernetes-on-your-machine) | 3 pods in a local Kubernetes cluster | pod in the cluster | mock (no key needed) | `http://localhost:18080` |
| [4. AWS by hand](#4-aws-by-hand-eksctl--the-aws-cli) | 2–6 pods on EKS, every resource created with `eksctl` / `aws` | RDS | Groq (demo) or a model in the VPC | `https://<your domain>` |
| [5. AWS with Terraform](#5-aws-with-terraform) | the same, the infrastructure as code (`deploy/terraform/`) | RDS | Groq (demo) or vLLM on a GPU node | `https://<your domain>` |

What each stage adds: 1 the app itself; 2 containers; 3 Kubernetes and several replicas; 4 the cloud pieces (managed database, load balancer, TLS, registry); 5 infrastructure as code.

The API is the same in every stage, so the [curl walkthrough in the README](../README.md#walkthrough-curl-commands) works against any of them; change `BASE` to the stage's URL.

---

## 0. One-time setup

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker with Compose, and Node.js/npm (for building two of the MCP servers). Stage 3 also needs [kind](https://kind.sigs.k8s.io/) and `kubectl`; stage 4 the `aws` CLI, [eksctl](https://eksctl.io) and [helm](https://helm.sh); stage 5 the `aws` CLI and Terraform ≥ 1.11.

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
make agent-eval                  # agent eval report (args="--category safety" for a quick run)
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

## 4. AWS by hand: eksctl + the AWS CLI

The kind manifests on real AWS, with every piece created by one command you type, so you can see what each piece is and why it's there. Stage 5 creates the same pieces with Terraform. [deployment.md](deployment.md) §10 explains the design.

What changes relative to kind:

| kind | AWS |
|---|---|
| Postgres pod | **RDS** (managed, encrypted, backed up), reachable only from the nodes |
| NodePort on localhost | **ALB** created from the Ingress by the AWS Load Balancer Controller, HTTPS with an **ACM** certificate |
| images loaded into kind | images pushed to **ECR** |
| mock LLM | a real model: Groq for a demo with synthetic data, or a model you host in the VPC |
| Secret from a local file | the same `kubectl create secret`, with values for RDS |

**Cost:** about $250–280 a month while it's up, mostly the EKS control plane, two nodes and the NAT gateway. Tear it down at the end (step 11).

Extra tools: the `aws` CLI with admin credentials (`aws sts get-caller-identity` works), [eksctl](https://eksctl.io), [helm](https://helm.sh), and a domain whose DNS you control.

**0. Settings used by every step** (one shell session):

```bash
export AWS_REGION=us-east-1
export CLUSTER=medagent-manual
export DOMAIN=medagent-demo.example.org            # yours
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
export VERSION=$(git rev-parse --short HEAD)
export GROQ_API_KEY=...                            # demo with Groq (synthetic data); empty with your own model
```

**1. Image repositories (ECR)**

```bash
for image in medagent medical-mcp-bridge healthcare-mcp research-mcp; do
  aws ecr create-repository --repository-name $image \
    --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true
done
```

**2. The cluster (EKS), about 20 minutes.** eksctl creates the VPC, the cluster, two private nodes, NetworkPolicy enforcement and the load balancer controller's IAM role (read `deploy/eksctl/cluster.example.yaml`: each line has a comment).

```bash
cp deploy/eksctl/cluster.example.yaml deploy/eksctl/cluster.yaml   # gitignored
# edit: your IP in publicAccessCIDRs (curl -s https://checkip.amazonaws.com)
eksctl create cluster -f deploy/eksctl/cluster.yaml                 # also points kubectl at it
kubectl get nodes                                                   # 2 nodes, Ready
```

**3. The load balancer controller**, which turns the Ingress into an ALB:

```bash
VPC_ID=$(aws eks describe-cluster --name $CLUSTER --query cluster.resourcesVpcConfig.vpcId --output text)
helm repo add eks https://aws.github.io/eks-charts && helm repo update
helm install aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system \
  --version 3.6.3 --set clusterName=$CLUSTER --set region=$AWS_REGION --set vpcId=$VPC_ID \
  --set serviceAccount.name=aws-load-balancer-controller
kubectl -n kube-system rollout status deployment/aws-load-balancer-controller
```

**4. The database (RDS for PostgreSQL 16), about 10 minutes.** It goes in the cluster's private subnets, and only the nodes may connect:

```bash
SUBNETS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID Name=tag:kubernetes.io/role/internal-elb,Values=1 \
  --query 'Subnets[].SubnetId' --output text)
aws rds create-db-subnet-group --db-subnet-group-name $CLUSTER \
  --db-subnet-group-description "MedAgent database" --subnet-ids $SUBNETS

DB_SG=$(aws ec2 create-security-group --group-name $CLUSTER-db --description "Postgres from EKS nodes" \
  --vpc-id $VPC_ID --query GroupId --output text)
NODE_SG=$(aws eks describe-cluster --name $CLUSTER --query cluster.resourcesVpcConfig.clusterSecurityGroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $DB_SG --protocol tcp --port 5432 --source-group $NODE_SG

PG_VERSION=$(aws rds describe-db-engine-versions --engine postgres \
  --query "DBEngineVersions[?starts_with(EngineVersion,'16.')].EngineVersion | [-1]" --output text)
DB_PASSWORD=$(openssl rand -hex 24)        # kept in this shell only; it goes into the Secret in step 6
aws rds create-db-instance --db-instance-identifier $CLUSTER \
  --engine postgres --engine-version $PG_VERSION --db-instance-class db.t4g.small \
  --allocated-storage 20 --storage-encrypted --backup-retention-period 7 \
  --db-name medagent --master-username medagent --master-user-password "$DB_PASSWORD" \
  --db-subnet-group-name $CLUSTER --vpc-security-group-ids $DB_SG --no-publicly-accessible
aws rds wait db-instance-available --db-instance-identifier $CLUSTER
DB_HOST=$(aws rds describe-db-instances --db-instance-identifier $CLUSTER \
  --query 'DBInstances[0].Endpoint.Address' --output text)
```

RDS for PostgreSQL 16 requires TLS by default (`rds.force_ssl=1`). The app's first migration runs `CREATE EXTENSION vector` as `medagent`, the master user, which is allowed to.

**5. The certificate (ACM).** Request it, add the record it prints at your DNS provider, and wait until it's issued:

```bash
CERT_ARN=$(aws acm request-certificate --domain-name $DOMAIN --validation-method DNS \
  --query CertificateArn --output text)
sleep 10
aws acm describe-certificate --certificate-arn $CERT_ARN \
  --query 'Certificate.DomainValidationOptions[0].ResourceRecord'      # add this CNAME
aws acm wait certificate-validated --certificate-arn $CERT_ARN         # minutes after the record exists
```

**6. The Secret.** The keys are in `deploy/k8s/base/secret.example.yaml`. Values with real secrets never go into a file in the repo:

```bash
kubectl create namespace medagent
kubectl -n medagent create secret generic medagent-secrets \
  --from-literal=POSTGRES_DSN="postgresql://medagent:$DB_PASSWORD@$DB_HOST:5432/medagent?sslmode=require" \
  --from-literal=JWT_SECRET_KEY="$(openssl rand -hex 32)" \
  --from-literal=ADMIN_BOOTSTRAP_USERNAME=admin \
  --from-literal=ADMIN_BOOTSTRAP_PASSWORD="$(openssl rand -hex 12 | tee /dev/stderr)" \
  --from-literal=GROQ_API_KEY="$GROQ_API_KEY" --from-literal=LLM_API_KEY=""
# the line on stderr is the bootstrap admin password: keep it for step 10
```

Another route is to sync the Secret from AWS Secrets Manager with `overlays/aws/external-secret.example.yaml`, which stage 5 does for you.

**7. Fill in every `<...>` placeholder** in `deploy/k8s/overlays/aws/kustomization.yaml` and `deploy/k8s/components/aws/ingress.yaml`:

| Placeholder | Value |
|---|---|
| image URIs and tag | `$REGISTRY/<image>`, `$VERSION` |
| `FORWARDED_ALLOW_IPS` | the VPC range, `10.30.0.0/16` (only the ALB may set the client IP) |
| `LLM_BASE_URL`, `LLM_MODEL` | your model in the VPC. For a demo with Groq instead, set `LLM_PROVIDER: groq`, remove those two lines, and add `ALLOW_EXTERNAL_LLM_WITH_PHI: "true"`, which here means **synthetic patients only** (production otherwise refuses a third-party model) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | your collector, or delete the line |
| `certificate-arn`, `host` | `$CERT_ARN`, `$DOMAIN` |

```bash
grep -rn '<' deploy/k8s/overlays/aws/kustomization.yaml deploy/k8s/components/aws/   # nothing left means done
kubectl kustomize deploy/k8s/overlays/aws > /dev/null                                # it builds
```

Don't commit these edits: they hold your account's values.

**8. Build, push, deploy**

```bash
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REGISTRY
docker compose build medagent medical-mcp-bridge healthcare-mcp research-mcp
for image in medagent medical-mcp-bridge healthcare-mcp research-mcp; do
  docker tag "medagent-$image:latest" "$REGISTRY/$image:$VERSION"
  docker push "$REGISTRY/$image:$VERSION"
done

kubectl apply -k deploy/k8s/overlays/aws
kubectl -n medagent rollout status deployment/medagent
kubectl -n medagent get pods,ingress            # ADDRESS: the ALB, after a minute or two
```

**9. DNS and checks.** Point `$DOMAIN` at the ALB address (a CNAME, or a Route 53 alias), then:

```bash
curl -s https://$DOMAIN/health
curl -s https://$DOMAIN/ready                   # the ALB health check uses this too
curl -s -o /dev/null -w '%{http_code}\n' https://$DOMAIN/metrics   # 404: not public
```

**10. First login.** Log in as `admin` with the password from step 6, create named admin and doctor accounts, then disable the bootstrap account ([deployment.md](deployment.md) §8). The README walkthrough works with `BASE=https://$DOMAIN`.

**Add a guideline**: the database isn't reachable from outside the VPC, so copy the file into an app pod (its `/tmp` is writable) and run the CLI there:

```bash
POD=$(kubectl -n medagent get pods -l app.kubernetes.io/name=medagent -o name | head -1 | cut -d/ -f2)
kubectl -n medagent cp docs/demo/synthetic-t2dm-guideline.txt $POD:/tmp/guideline.txt -c medagent
kubectl -n medagent exec $POD -c medagent -- \
  python -m medagent.cli ingest-guideline --file /tmp/guideline.txt --source synthetic-t2dm
```

Every pod finds it, because the guidelines are in Postgres.

**Updating to a new version:** build and push with a new `VERSION`, change the tag in `kustomization.yaml`, then run `kubectl apply -k deploy/k8s/overlays/aws` again.
- Pods are replaced one at a time (`maxUnavailable: 0`).
- Each new pod's init container applies any new migrations first.
- A doctor whose WebSocket is cut reconnects with `?resume=1` and gets their draft back.

**11. Tear down.** Do this the same day you finish; every hour costs money. The order matters: each step removes something the next one would otherwise be blocked by, or would orphan.

```bash
# a. The app. The load balancer controller then deletes the ALB in the background,
#    so wait for it; deleting the cluster first would orphan a billing ALB.
kubectl delete -k deploy/k8s/overlays/aws
make aws-wait-alb cluster=$CLUSTER

# b. The database: no final snapshot, and its automated backups go with it.
aws rds delete-db-instance --db-instance-identifier $CLUSTER \
  --skip-final-snapshot --delete-automated-backups
aws rds wait db-instance-deleted --db-instance-identifier $CLUSTER
aws rds delete-db-subnet-group --db-subnet-group-name $CLUSTER
aws ec2 delete-security-group --group-id $DB_SG        # only once RDS is gone

# c. The cluster: nodes, VPC, NAT gateway and its Elastic IP, IAM roles (~15 min).
eksctl delete cluster -f deploy/eksctl/cluster.yaml --wait
aws logs delete-log-group --log-group-name /aws/eks/$CLUSTER/cluster   # EKS leaves it behind

# d. Images and the certificate (no longer used by any ALB).
for image in medagent medical-mcp-bridge healthcare-mcp research-mcp; do
  aws ecr delete-repository --repository-name $image --force
done
aws acm delete-certificate --certificate-arn $CERT_ARN

# e. Check: lists anything still there and exits 1 if so.
make aws-leftovers cluster=$CLUSTER domain=$DOMAIN

git checkout deploy/k8s/overlays/aws deploy/k8s/components/aws   # drop your filled-in values
```

Then remove the DNS records you added (the CNAMEs for the ALB and the certificate) at your DNS provider.

- **A new shell lost the variables:** run step 0 again. `DB_SG` and `CERT_ARN` can be found again with `aws ec2 describe-security-groups --filters Name=group-name,Values=$CLUSTER-db` and `aws acm list-certificates`.
- **A step fails because something is still in use:** wait a minute and run it again. AWS deletes some things in the background, the load balancer's network interfaces for example. `make aws-leftovers` shows what's left.
- **Not tied to the cluster name:** `make aws-leftovers` also flags any unattached Elastic IP in the region. If it isn't yours from this stage, ignore it.

**What this stage teaches, and what it leaves to you:** every resource above exists because you typed it, and nothing records it. Re-creating the stack, changing it later or handing it to someone else means doing it again by hand. That's what stage 5 fixes.

**Before real patients**, also go through [deployment.md](deployment.md): §4 the model, §5 encryption and backups, §7 retention, §11 outbound traffic, §13 known limits.

---

## 5. AWS with Terraform

The same infrastructure as stage 4, written as code in `deploy/terraform/`:
- **Repeatable:** `terraform plan` shows what would change before anything does, and the same files rebuild the stack in another account or region.
- **Recorded:** the state, kept in S3, is the record of what exists.
- **Destroyed in one command.**
- **Secrets stay out of the repo and out of Terraform's state.**

`scripts/render_aws_overlay.sh` replaces stage 4's step 7: it writes `deploy/k8s/overlays/aws-generated/` from Terraform's outputs, with no placeholders to fill in.

| Stage 4 step | Terraform file |
|---|---|
| 1. ECR | `ecr.tf` |
| 2. cluster + VPC (eksctl) | `vpc.tf`, `eks.tf` |
| 3. load balancer controller | `addons.tf` (plus External Secrets, so step 6 is automatic) |
| 4. RDS | `rds.tf` |
| 5. ACM | `acm.tf` (validated for you with a Route 53 zone) |
| 6. Secret | `secrets.tf` (Secrets Manager, synced into the cluster) |
| 7. placeholders | `outputs.tf` + `make aws-overlay` |
| 11. tear down | `make tf-down` (destroy + the same leftovers check) |

It also has an optional GPU node running vLLM (`enable_gpu_model = true`), so prompts never leave your account.

Extra tools: Terraform ≥ 1.11. If you did stage 4, tear it down first (step 11).

```bash
# once per account: the S3 bucket for Terraform state
make tf-bootstrap AWS_REGION=us-east-1
export TF_STATE_BUCKET=medagent-tfstate-<account id>       # printed by tf-bootstrap

cp deploy/terraform/demo.tfvars.example deploy/terraform/demo.tfvars   # gitignored
# edit: domain_name, cluster_public_access_cidrs (your IP), optionally route53_zone_id
export TF_VAR_groq_api_key=$GROQ_API_KEY                   # demo without a GPU node

make tf-init TF_ENV=demo
make tf-plan TF_ENV=demo          # read it: every resource it will create
make tf-apply TF_ENV=demo         # ~20 minutes

terraform -chdir=deploy/terraform output acm_validation_records   # without Route 53: add these
$(terraform -chdir=deploy/terraform output -raw kubeconfig_command)

make aws-overlay version=$(git rev-parse --short HEAD)    # writes the overlay, prints the push commands
# ... run the printed docker login / build / push ...
kubectl apply -k deploy/k8s/overlays/aws-generated
kubectl -n medagent rollout status deployment/medagent
```

Then do stage 4's steps 9 and 10: DNS, checks and first login. The bootstrap password is in Secrets Manager (`deploy/terraform/README.md`).

**Updating:** push with a new version, then run `make aws-overlay version=...` and `kubectl apply -k deploy/k8s/overlays/aws-generated`.

**Changing the infrastructure:** edit a variable or a `.tf` file, then `make tf-plan` and `make tf-apply`.

**Tear down**, all in one command, in the order that leaves nothing billing:

```bash
make tf-down TF_ENV=demo domain=medagent-demo.example.org
```

1. `kubectl delete -k deploy/k8s/overlays/aws-generated`: the app, so the controller deletes the ALB.
2. It waits until the ALB and its security groups are gone. Destroying the controller first would orphan them.
3. `terraform destroy` removes everything Terraform created, including the RDS instance with its backups (no final snapshot in a demo), the ECR images, the secret (no recovery window in a demo), the certificate and the Route 53 validation records.
4. `make aws-leftovers` lists anything still there and fails if it finds something.

Then remove the DNS record for the ALB. If you're done with AWS for good, also remove the state bucket (`deploy/terraform/README.md`, *Tearing it down*).

A production stack (`deletion_protection = true`) refuses to destroy its database. That's the point of the setting. To really delete it, set `deletion_protection = false`, run `make tf-plan` and `make tf-apply`, then `make tf-down`; the final snapshot is then skipped too.

The full runbook (costs, the model choice, rotating secrets, what's not covered) is [deploy/terraform/README.md](../deploy/terraform/README.md).

---

## Quick reference

| Task | 1. Local | 2. Docker | 3. kind | 4. AWS by hand | 5. AWS + Terraform |
|---|---|---|---|---|---|
| create infra | — | — | `make k8s-local-up` (cluster) | `aws ecr ...`, `eksctl create cluster`, `aws rds ...`, `aws acm ...` | `make tf-init tf-plan tf-apply` |
| start the app | `make docker-up-deps` + `make serve` | `make docker-up` | `make k8s-local-up` | `kubectl apply -k deploy/k8s/overlays/aws` | `make aws-overlay version=...` + `kubectl apply -k deploy/k8s/overlays/aws-generated` |
| stop | Ctrl+C, `make docker-down` | `make docker-down` | `make k8s-local-down` | step 11, then `make aws-leftovers cluster=medagent-manual` | `make tf-down TF_ENV=demo` (includes the leftovers check) |
| migrations | `make db-upgrade` | on app startup | `migrate` init container | `migrate` init container | `migrate` init container |
| logs | terminal | `make docker-logs` | `kubectl -n medagent-local logs -f deploy/medagent` | `kubectl -n medagent logs -f deploy/medagent` | same as 4 |
| ingest a guideline | `make ingest-guideline ...` | `docker compose exec medagent python -m medagent.cli ingest-guideline ...` | CLI via `port-forward svc/postgres` | `kubectl cp` + `kubectl exec` into a pod | same as 4 |
| URL | `http://localhost:8000` | `http://localhost:8000` | `http://localhost:18080` | `https://<your domain>` | `https://<your domain>` |
