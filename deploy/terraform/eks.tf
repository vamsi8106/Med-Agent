locals {
  general_nodes = {
    general = {
      ami_type       = "AL2023_x86_64_STANDARD"
      instance_types = [var.node_instance_type]
      min_size       = var.node_count
      desired_size   = var.node_count
      max_size       = var.node_count * 2
      # IMDSv2 only, one hop: pods cannot borrow the node's IAM role.
      metadata_options = {
        http_endpoint               = "enabled"
        http_tokens                 = "required"
        http_put_response_hop_limit = 1
      }
    }
  }

  # Only GPU work lands here: the taint keeps everything else off, vLLM
  # tolerates it (deploy/k8s/components/vllm).
  gpu_nodes = var.enable_gpu_model ? {
    gpu = {
      ami_type       = "AL2023_x86_64_NVIDIA"
      instance_types = [var.gpu_instance_type]
      min_size       = 1
      desired_size   = 1
      max_size       = 1
      disk_size      = 100 # model weights + the vLLM image
      labels         = { "nvidia.com/gpu.present" = "true" }
      taints = {
        gpu = { key = "nvidia.com/gpu", value = "true", effect = "NO_SCHEDULE" }
      }
      metadata_options = {
        http_endpoint               = "enabled"
        http_tokens                 = "required"
        http_put_response_hop_limit = 1
      }
    }
  } : {}
}

# Accepted scanner findings (make tf-check), each on purpose:
# - AWS-0040: the API endpoint is public so kubectl and this Terraform work
#   without a VPN; AWS-0041 holds -- only cluster_public_access_cidrs may reach it.
# - AWS-0104: nodes need egress to the internet: image pulls (ECR, Docker Hub,
#   Hugging Face for vLLM) and the MCP servers' public medical APIs. Pod-level
#   egress is narrowed by the NetworkPolicies in deploy/k8s/base.
#trivy:ignore:AWS-0040
#trivy:ignore:AWS-0104
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "21.30.2"

  name               = local.name
  kubernetes_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # kubectl and this Terraform reach the API from outside; nodes use the
  # private endpoint.
  endpoint_public_access       = true
  endpoint_private_access      = true
  endpoint_public_access_cidrs = var.cluster_public_access_cidrs

  # Whoever runs the first apply becomes cluster admin (EKS access entry).
  enable_cluster_creator_admin_permissions = true

  # Kubernetes Secrets encrypted with a KMS key; control-plane audit logs.
  enabled_log_types = ["api", "audit", "authenticator"]

  addons = {
    coredns    = {}
    kube-proxy = {}
    vpc-cni = {
      before_compute = true
      # The base manifests rely on NetworkPolicies (deploy/k8s/base/network-policies.yaml).
      configuration_values = jsonencode({ enableNetworkPolicy = "true" })
    }
    eks-pod-identity-agent = { before_compute = true }
  }

  eks_managed_node_groups = merge(local.general_nodes, local.gpu_nodes)
}
