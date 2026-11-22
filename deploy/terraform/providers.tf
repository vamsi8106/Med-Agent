provider "aws" {
  region = var.region
  default_tags {
    tags = { project = "medagent", env = var.env, managed-by = "terraform" }
  }
}

# Both talk to the cluster this root creates, authenticating through the AWS
# CLI (a short-lived token per call, nothing stored).
locals {
  cluster_exec = {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", var.region]
  }
}

provider "helm" {
  kubernetes = {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    exec                   = local.cluster_exec
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  exec {
    api_version = local.cluster_exec.api_version
    command     = local.cluster_exec.command
    args        = local.cluster_exec.args
  }
}
