variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "env" {
  description = "Environment name: demo or production. Names resources and the secret (medagent/<env>)."
  type        = string
  default     = "demo"
  validation {
    condition     = contains(["demo", "production"], var.env)
    error_message = "env must be demo or production."
  }
}

variable "domain_name" {
  description = "Host name doctors use, e.g. medagent.example.org. Gets the ACM certificate and the ingress host."
  type        = string
}

variable "route53_zone_id" {
  description = "Route 53 hosted zone for domain_name. Set: the certificate is validated automatically. Empty: add the records from the acm_validation_records output yourself."
  type        = string
  default     = ""
}

variable "vpc_cidr" {
  description = "VPC address range. Also FORWARDED_ALLOW_IPS: only the ALB, inside it, may set X-Forwarded-For."
  type        = string
  default     = "10.20.0.0/16"
}

variable "single_nat_gateway" {
  description = "One NAT gateway for the whole VPC (cheaper) instead of one per AZ (survives an AZ outage)."
  type        = bool
  default     = true
}

variable "kubernetes_version" {
  description = "EKS Kubernetes version."
  type        = string
  default     = "1.36"
}

variable "cluster_public_access_cidrs" {
  description = "Who may reach the Kubernetes API from the internet (kubectl, this Terraform): your IP, office or VPN, e.g. [\"203.0.113.10/32\"]."
  type        = list(string)
  validation {
    condition     = length(var.cluster_public_access_cidrs) > 0 && !contains(var.cluster_public_access_cidrs, "0.0.0.0/0")
    error_message = "List the addresses that may reach the Kubernetes API; 0.0.0.0/0 opens it to the internet."
  }
}

variable "node_instance_type" {
  description = "Instance type of the general node group (app, MCP servers, controllers)."
  type        = string
  default     = "t3.large"
}

variable "node_count" {
  description = "Nodes in the general group: desired and minimum (the maximum is twice this)."
  type        = number
  default     = 2
}

variable "db_instance_class" {
  description = "RDS instance class."
  type        = string
  default     = "db.t4g.small"
}

variable "db_allocated_storage" {
  description = "RDS storage in GiB (grows automatically up to 5x)."
  type        = number
  default     = 20
}

variable "db_multi_az" {
  description = "A standby RDS instance in a second AZ."
  type        = bool
  default     = false
}

variable "db_backup_retention_days" {
  description = "Days of automated RDS backups (point-in-time restore)."
  type        = number
  default     = 7
}

variable "deletion_protection" {
  description = "Protect RDS from deletion, take a final snapshot, keep ECR images and the secret's recovery window. Off for a demo you will destroy."
  type        = bool
  default     = false
}

variable "enable_gpu_model" {
  description = "Create a GPU node group and run the model in the cluster (vLLM), so prompts with patient data stay in this account. Costs about USD 1/hour per g5.xlarge while up."
  type        = bool
  default     = false
}

variable "gpu_instance_type" {
  description = "Instance type of the GPU node group."
  type        = string
  default     = "g5.xlarge"
}

variable "model_name" {
  description = "Hugging Face model vLLM serves. Must support tool calling; run make eval against it before real use."
  type        = string
  default     = "Qwen/Qwen2.5-7B-Instruct"
}

variable "secrets_version" {
  description = "Bump to generate new database, JWT and bootstrap-admin passwords and write them again (rotation). Also rewrites the API keys from the variables below."
  type        = number
  default     = 1
}

variable "groq_api_key" {
  description = "Optional, for a demo with synthetic data. Written to the secret only when secrets_version changes. Set with TF_VAR_groq_api_key, never in a tfvars file."
  type        = string
  default     = ""
  sensitive   = true
  ephemeral   = true
}

variable "llm_api_key" {
  description = "Optional key for an openai_compatible endpoint. Same handling as groq_api_key."
  type        = string
  default     = ""
  sensitive   = true
  ephemeral   = true
}
