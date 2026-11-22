# scripts/render_aws_overlay.sh reads these (terraform output -json).

output "region" {
  value = var.region
}

output "env" {
  value = var.env
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "kubeconfig_command" {
  description = "Run once to point kubectl at the cluster."
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}

output "registry" {
  description = "ECR registry host; images are <registry>/<name>:<version>."
  value       = split("/", aws_ecr_repository.image["medagent"].repository_url)[0]
}

output "images" {
  value = local.images
}

output "vpc_cidr" {
  value = module.vpc.vpc_cidr_block
}

output "domain_name" {
  value = var.domain_name
}

output "certificate_arn" {
  value = aws_acm_certificate.app.arn
}

output "acm_validation_records" {
  description = "Add these at your DNS provider when route53_zone_id is empty."
  value       = local.validation_records
}

output "secret_name" {
  value = aws_secretsmanager_secret.app.name
}

output "enable_gpu_model" {
  value = var.enable_gpu_model
}

output "model_name" {
  value = var.model_name
}

output "db_address" {
  description = "RDS host, reachable only from inside the VPC."
  value       = module.db.db_instance_address
}
