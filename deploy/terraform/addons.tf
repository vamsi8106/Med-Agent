# Cluster add-ons the AWS overlay needs, each with an IAM role through EKS Pod
# Identity (no long-lived keys, no IRSA OIDC plumbing).

module "lb_controller_identity" {
  source  = "terraform-aws-modules/eks-pod-identity/aws"
  version = "2.11.0"

  name                            = "${local.name}-aws-lb-controller"
  attach_aws_lb_controller_policy = true
  associations = {
    this = {
      cluster_name    = module.eks.cluster_name
      namespace       = "kube-system"
      service_account = "aws-load-balancer-controller"
    }
  }
}

resource "helm_release" "lb_controller" {
  name       = "aws-load-balancer-controller"
  namespace  = "kube-system"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  version    = "3.6.3"

  set = [
    { name = "clusterName", value = module.eks.cluster_name },
    { name = "region", value = var.region },
    { name = "vpcId", value = module.vpc.vpc_id },
    { name = "serviceAccount.name", value = "aws-load-balancer-controller" },
  ]

  depends_on = [module.eks, module.lb_controller_identity]
}

# Reads exactly one secret, medagent/<env>, never writes.
module "external_secrets_identity" {
  source  = "terraform-aws-modules/eks-pod-identity/aws"
  version = "2.11.0"

  name                                  = "${local.name}-external-secrets"
  attach_external_secrets_policy        = true
  external_secrets_secrets_manager_arns = [aws_secretsmanager_secret.app.arn]
  external_secrets_create_permission    = false
  associations = {
    this = {
      cluster_name    = module.eks.cluster_name
      namespace       = "external-secrets"
      service_account = "external-secrets"
    }
  }
}

resource "helm_release" "external_secrets" {
  name             = "external-secrets"
  namespace        = "external-secrets"
  create_namespace = true
  repository       = "https://charts.external-secrets.io"
  chart            = "external-secrets"
  version          = "2.14.0"

  set = [
    { name = "serviceAccount.name", value = "external-secrets" },
    { name = "installCRDs", value = "true" },
  ]

  depends_on = [module.eks, module.external_secrets_identity]
}

# Exposes the GPU to pods (nvidia.com/gpu); the NVIDIA AMI has the driver.
resource "helm_release" "nvidia_device_plugin" {
  count = var.enable_gpu_model ? 1 : 0

  name       = "nvidia-device-plugin"
  namespace  = "kube-system"
  repository = "https://nvidia.github.io/k8s-device-plugin"
  chart      = "nvidia-device-plugin"
  version    = "0.20.1"

  values = [yamlencode({
    nodeSelector = { "nvidia.com/gpu.present" = "true" }
    tolerations  = [{ key = "nvidia.com/gpu", operator = "Exists", effect = "NoSchedule" }]
  })]

  depends_on = [module.eks]
}
