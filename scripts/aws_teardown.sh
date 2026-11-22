#!/usr/bin/env bash
# Helpers so tearing down AWS leaves nothing billing behind.
#
#   scripts/aws_teardown.sh wait-alb  <cluster>            # after `kubectl delete -k ...`
#   scripts/aws_teardown.sh leftovers <cluster> [domain]   # after everything is deleted
#
# <cluster> is the EKS cluster name: medagent-manual (docs/running.md stage 4)
# or medagent-<env> (stage 5, Terraform). The RDS instance has the same name
# in both. Uses AWS_REGION (default us-east-1).
#
# wait-alb: the AWS Load Balancer Controller deletes the ALB and its security
# groups *after* the Ingress is gone, in the background. Deleting the cluster
# first orphans them (an ALB bills by the hour) and blocks the VPC deletion.
# This waits until the controller has removed both.
#
# leftovers: lists every billable thing still tied to the cluster name, and
# exits 1 if it finds any. Run it last; an empty report means the bill stops.
set -euo pipefail

cmd="${1:?usage: $0 wait-alb|leftovers <cluster> [domain]}"
cluster="${2:?usage: $0 $cmd <cluster> [domain]}"
domain="${3:-}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_PAGER=""

# Resources the load balancer controller creates carry this tag.
alb_tag="elbv2.k8s.aws/cluster"

albs() {
  aws resourcegroupstaggingapi get-resources \
    --resource-type-filters elasticloadbalancing:loadbalancer \
    --tag-filters "Key=$alb_tag,Values=$cluster" \
    --query 'ResourceTagMappingList[].ResourceARN' --output text
}

alb_security_groups() {
  aws ec2 describe-security-groups --filters "Name=tag:$alb_tag,Values=$cluster" \
    --query 'SecurityGroups[].GroupId' --output text
}

wait_alb() {
  local deadline=$((SECONDS + 600))
  while :; do
    local lb sg
    lb="$(albs)"
    sg="$(alb_security_groups)"
    if [[ -z "$lb" && -z "$sg" ]]; then
      echo "load balancer and its security groups are gone"
      return 0
    fi
    if ((SECONDS > deadline)); then
      echo "still there after 10 minutes -- delete them in the console (EC2 > Load Balancers / Security Groups):" >&2
      echo "  load balancers: ${lb:-none}" >&2
      echo "  security groups: ${sg:-none}" >&2
      return 1
    fi
    echo "waiting for the controller to delete the load balancer..."
    sleep 15
  done
}

found=0
report() { # report <what> <ids...>: print and count, if any ids
  local what="$1"
  shift
  local ids="$*"
  if [[ -n "${ids// /}" && "$ids" != "None" ]]; then
    printf '  LEFT  %-34s %s\n' "$what" "$ids"
    found=1
  else
    printf '  ok    %s\n' "$what"
  fi
}

leftovers() {
  echo "Billable resources still tied to '$cluster' in $AWS_REGION:"

  report "EKS cluster" "$(aws eks list-clusters --query "clusters[?@=='$cluster']" --output text)"
  report "load balancers (ALB)" "$(albs)"
  report "ALB security groups" "$(alb_security_groups)"

  report "RDS instance" "$(aws rds describe-db-instances \
    --query "DBInstances[?DBInstanceIdentifier=='$cluster'].DBInstanceIdentifier" --output text)"
  report "RDS snapshots" "$(aws rds describe-db-snapshots --db-instance-identifier "$cluster" \
    --query 'DBSnapshots[].DBSnapshotIdentifier' --output text 2>/dev/null || true)"
  report "RDS retained backups" "$(aws rds describe-db-instance-automated-backups --db-instance-identifier "$cluster" \
    --query 'DBInstanceAutomatedBackups[].DbiResourceId' --output text 2>/dev/null || true)"

  # The VPC: eksctl tags it with the cluster name, Terraform names it after it.
  local vpcs
  vpcs="$(aws ec2 describe-vpcs \
    --filters "Name=tag:alpha.eksctl.io/cluster-name,Values=$cluster" \
    --query 'Vpcs[].VpcId' --output text) $(aws ec2 describe-vpcs \
    --filters "Name=tag:Name,Values=$cluster" --query 'Vpcs[].VpcId' --output text)"
  vpcs="$(tr -s ' \t' '\n' <<<"$vpcs" | sort -u | tr '\n' ' ')"
  report "VPC" "$vpcs"
  local nat=""
  for vpc in $vpcs; do
    nat+=" $(aws ec2 describe-nat-gateways --filter "Name=vpc-id,Values=$vpc" \
      "Name=state,Values=pending,available,deleting" --query 'NatGateways[].NatGatewayId' --output text)"
  done
  report "NAT gateways" "$nat"
  # Not tied to a name: any unattached Elastic IP in the region bills.
  report "unattached Elastic IPs (any)" "$(aws ec2 describe-addresses \
    --query 'Addresses[?AssociationId==null].PublicIp' --output text)"
  report "unattached EBS volumes" "$(aws ec2 describe-volumes \
    --filters Name=status,Values=available "Name=tag-key,Values=kubernetes.io/cluster/$cluster" \
    --query 'Volumes[].VolumeId' --output text)"

  report "ECR repositories" "$(aws ecr describe-repositories \
    --query "repositories[?contains(['medagent','medical-mcp-bridge','healthcare-mcp','research-mcp'], repositoryName)].repositoryName" \
    --output text)"
  if [[ -n "$domain" ]]; then
    report "ACM certificate ($domain)" "$(aws acm list-certificates \
      --query "CertificateSummaryList[?DomainName=='$domain'].CertificateArn" --output text)"
  fi
  report "Secrets Manager (medagent/*)" "$(aws secretsmanager list-secrets \
    --filters Key=name,Values=medagent/ --query 'SecretList[].Name' --output text)"
  report "log groups (/aws/eks, /aws/rds)" "$(aws logs describe-log-groups --log-group-name-prefix "/aws/eks/$cluster" \
    --query 'logGroups[].logGroupName' --output text) $(aws logs describe-log-groups \
    --log-group-name-prefix "/aws/rds/instance/$cluster" --query 'logGroups[].logGroupName' --output text)"
  report "eksctl CloudFormation stacks" "$(aws cloudformation list-stacks \
    --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE ROLLBACK_COMPLETE DELETE_FAILED CREATE_IN_PROGRESS DELETE_IN_PROGRESS \
    --query "StackSummaries[?starts_with(StackName, 'eksctl-$cluster-')].StackName" --output text)"
  report "KMS keys not yet scheduled for deletion" "$(for key in $(aws kms list-aliases \
    --query "Aliases[?starts_with(AliasName, 'alias/eks/$cluster')].TargetKeyId" --output text); do
    aws kms describe-key --key-id "$key" --query "KeyMetadata.KeyState=='Enabled' && KeyMetadata.KeyId || ''" --output text
  done)"

  echo
  if ((found)); then
    echo "Something is still there (LEFT above): delete it, then run this again."
    echo "Also remove the DNS record you added for the domain, if any."
    return 1
  fi
  echo "Nothing left: the bill for '$cluster' has stopped."
  echo "Also remove the DNS record you added for the domain, if any."
}

case "$cmd" in
  wait-alb) wait_alb ;;
  leftovers) leftovers ;;
  *) echo "unknown command: $cmd (wait-alb | leftovers)" >&2; exit 2 ;;
esac
