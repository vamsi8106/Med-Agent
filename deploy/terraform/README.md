# AWS infrastructure (Terraform)

Stage 5 of [docs/running.md](../../docs/running.md#5-aws-with-terraform): everything the AWS
deployment (`deploy/k8s/`) expects, created in an empty account. Stage 4
creates the same pieces by hand with eksctl and the AWS CLI; do that first if
you want to see what each one is. After `make tf-apply`, deploying the app is pushing the images and
running one `kubectl apply`.

| File | Creates |
|---|---|
| `bootstrap/` | the S3 bucket for Terraform state (versioned, private, TLS only, its own KMS key), once per account |
| `vpc.tf` | VPC over 2 AZs: public subnets (ALB), private subnets (nodes, pods), database subnets; NAT; rejected-traffic flow logs |
| `eks.tf` | EKS with private nodes (IMDSv2 only), VPC CNI with NetworkPolicy enforcement, Pod Identity, audit logs; an optional GPU node group (tainted, only vLLM runs there) |
| `rds.tf` | PostgreSQL 16 (pgvector), encrypted, backups, TLS forced, reachable only from the nodes |
| `ecr.tf` | the four image repositories: immutable tags, scan on push, last 20 images kept |
| `acm.tf` | the TLS certificate; validated automatically with a Route 53 zone |
| `secrets.tf` | Secrets Manager `medagent/<env>`: `POSTGRES_DSN`, `JWT_SECRET_KEY`, the bootstrap admin, API keys |
| `addons.tf` | AWS Load Balancer Controller, External Secrets Operator, NVIDIA device plugin (GPU only), each with its own IAM role |

## What it costs

Rough on-demand prices in us-east-1, while it is up. Check the AWS pricing
calculator for your region.

| | Demo (`demo.tfvars.example`) | Production (`production.tfvars.example`) |
|---|---|---|
| EKS control plane | ~$73 | ~$73 |
| Nodes | 2 × t3.large, ~$120 | 3 × m6i.large, ~$210 |
| RDS | db.t4g.small single-AZ, ~$25 | db.m7g.large Multi-AZ, ~$250 |
| NAT gateway | 1, ~$35 + traffic | 2, ~$70 + traffic |
| ALB, logs, secrets, KMS | ~$25 | ~$30 |
| GPU node (g5.xlarge) | off (Groq) | ~$730 (~$1/hour) |
| **Total / month** | **~$250–280** | **~$1,350+** |

`make tf-down` removes a demo completely and checks that nothing is left
(*Tearing it down*, below).

## Secrets: none in state
Passwords are generated as ephemeral values and written through write-only
arguments, to RDS and to the Secrets Manager secret. Terraform state never
holds them. They're written only when `secrets_version` changes. To rotate
the database, JWT and bootstrap-admin passwords together, bump it and apply.
That signs everyone out.

API keys follow the same rule. Pass them as environment variables at the
first apply, or at a rotation, never in a tfvars file:

```bash
export TF_VAR_groq_api_key=...     # demo with synthetic data only
```

To change only an API key later, without rotating everything, edit the
secret in the AWS console or with `aws secretsmanager put-secret-value`.
External Secrets syncs it within an hour, or right away with
`kubectl -n medagent annotate externalsecret medagent-secrets force-sync=$(date +%s) --overwrite`,
then `kubectl -n medagent rollout restart deployment/medagent`.

## The model
- **`enable_gpu_model = false` (demo default):**
  - In a demo, the app uses Groq. `scripts/render_aws_overlay.sh` sets `ALLOW_EXTERNAL_LLM_WITH_PHI=true`, which here means *only synthetic patients are entered*. Never put real patient data into a demo.
  - In production, the renderer refuses this combination.
- **`enable_gpu_model = true`:** a g5.xlarge node runs vLLM serving `model_name` (`deploy/k8s/components/vllm`), and prompts never leave the cluster.
  - The default model is Qwen2.5-7B-Instruct. Run `make eval` against it, or any other model, before relying on its answers.
  - The first start downloads the weights, which takes several minutes.
- **A model elsewhere in the VPC:** leave the GPU group off and run `LLM_BASE_URL=http://10.20.x.x:8000/v1 make aws-overlay version=...`.

## Steps

Prerequisites: an AWS account, credentials with admin rights (`aws sts
get-caller-identity` works), Terraform ≥ 1.11, kubectl, Docker and jq.

```bash
# 1. Once per account: the state bucket (prints its name)
make tf-bootstrap AWS_REGION=us-east-1
export TF_STATE_BUCKET=medagent-tfstate-<account id>

# 2. Variables
cp deploy/terraform/demo.tfvars.example deploy/terraform/demo.tfvars    # gitignored
#    set domain_name and cluster_public_access_cidrs (your IP), optionally route53_zone_id
export TF_VAR_groq_api_key=...                                          # demo without a GPU

# 3. Create it (~20 minutes, EKS and RDS take longest)
make tf-init TF_ENV=demo
make tf-plan TF_ENV=demo          # read the plan
make tf-apply TF_ENV=demo

# 4. Without a Route 53 zone: add the certificate's validation records at your DNS provider
terraform -chdir=deploy/terraform output acm_validation_records

# 5. kubectl
$(terraform -chdir=deploy/terraform output -raw kubeconfig_command)

# 6. The overlay, then build, push and deploy (the script prints these commands)
VERSION=$(git rev-parse --short HEAD)
make aws-overlay version=$VERSION
#    ... run the printed docker login / build / push commands ...
kubectl apply -k deploy/k8s/overlays/aws-generated
kubectl -n medagent rollout status deployment/medagent

# 7. DNS: point domain_name at the ALB
kubectl -n medagent get ingress medagent      # ADDRESS column
curl -s https://<domain_name>/ready
```

Then do the first login: `docs/running.md` stage 4, step 10. The bootstrap admin's
password is in the secret:

```bash
aws secretsmanager get-secret-value --secret-id medagent/demo --query SecretString --output text | jq -r .ADMIN_BOOTSTRAP_PASSWORD
```

**Updating the app:** push the images with a new `VERSION`, then run `make
aws-overlay version=$VERSION` and `kubectl apply -k
deploy/k8s/overlays/aws-generated`.

## Tearing it down

```bash
make tf-down TF_ENV=demo domain=<domain_name>
```

This runs the steps in an order that leaves nothing billing:
1. **The app:** `kubectl delete -k deploy/k8s/overlays/aws-generated`, so the load balancer controller deletes the ALB.
2. **Wait:** `scripts/aws_teardown.sh wait-alb` waits until the ALB and its security groups are gone. If `terraform destroy` removed the controller first, they'd be orphaned and keep billing, and the VPC deletion would fail.
3. **`terraform destroy`:** everything this root created.
   - With `deletion_protection = false` (demo), the database goes with no final snapshot and with its automated backups, ECR repositories are deleted with their images, and the secret is deleted with no recovery window.
   - KMS keys are scheduled for deletion (AWS's minimum wait is 7 days).
4. **Check:** `scripts/aws_teardown.sh leftovers` lists anything still tied to `medagent-<env>` and exits 1 if it finds something. That covers the EKS cluster, ALB and security groups, RDS instance, snapshots and backups, VPC, NAT gateways, unattached Elastic IPs and EBS volumes, ECR, the certificate, `medagent/*` secrets, log groups, eksctl stacks and enabled KMS keys.

Afterwards, remove the DNS record for the ALB at your DNS provider.

**Production (`deletion_protection = true`):** the database refuses to be destroyed. That's the point of the setting. To really delete it, set `deletion_protection = false`, run `make tf-plan` and `make tf-apply`, then `make tf-down`.

**Finished with AWS for good:** the state bucket and its KMS key are kept, with `prevent_destroy`, because they hold the record of what exists. Together they cost about $1 a month. Remove them only after `make tf-down` reports nothing left, for every environment:

```bash
B=$TF_STATE_BUCKET
aws s3api delete-objects --bucket $B --delete "$(aws s3api list-object-versions --bucket $B \
  --query '{Objects: [Versions, DeleteMarkers][][].{Key: Key, VersionId: VersionId}}' --output json)"
aws s3api delete-bucket --bucket $B
KEY=$(aws kms describe-key --key-id alias/medagent-tfstate --query KeyMetadata.KeyId --output text)
aws kms delete-alias --alias-name alias/medagent-tfstate
aws kms schedule-key-deletion --key-id $KEY --pending-window-in-days 7
rm -f deploy/terraform/bootstrap/terraform.tfstate*
```

## Checks (offline, in CI)

- `make tf-check`: `terraform fmt`, `terraform validate` for both roots, and a
  Trivy scan that fails on high or critical findings. The findings accepted on
  purpose are written down above the `module "eks"` block in `eks.tf`.
- `make aws-overlay-check`: renders the overlay from
  `testdata/outputs-*.json` (a Groq demo and production with vLLM) and checks
  that the manifests build with no placeholder left.

Neither needs AWS credentials. A real `plan` or `apply` is never run in CI.

## Not covered

- **DNS for the app:** a Route 53 alias or CNAME to the ALB. The ALB is
  created by the controller after deploy, so it isn't known to this root.
- **Monitoring backends:** Amazon Managed Prometheus, Grafana, an OTel
  collector. Pass `OTEL_EXPORTER_OTLP_ENDPOINT` to the renderer once you have
  one.
- **WAF:** set the annotation in `deploy/k8s/components/aws/ingress.yaml`.
- **A dedicated database user:** the app connects as the RDS master user,
  which the pgvector migration needs.
