# Run once per AWS account, before the main root: the S3 bucket that holds
# Terraform state. Its own state stays local (it is one bucket); keep the
# terraform.tfstate it writes, or import the bucket if you lose it.
#
# The main root's state holds no passwords (they are ephemeral and
# write-only, see ../secrets.tf), but it does describe the whole deployment,
# so the bucket is private, encrypted and versioned.

terraform {
  required_version = ">= 1.11"
  required_providers {
    aws = { source = "hashicorp/aws", version = "= 6.73.0" }
  }
}

variable "region" {
  description = "AWS region for the state bucket (use the same as the deployment)."
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "Globally unique bucket name. Empty: medagent-tfstate-<account id>."
  type        = string
  default     = ""
}

provider "aws" {
  region = var.region
  default_tags { tags = { project = "medagent", managed-by = "terraform" } }
}

data "aws_caller_identity" "current" {}

locals {
  bucket = var.bucket_name != "" ? var.bucket_name : "medagent-tfstate-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "state" {
  bucket = local.bucket
  # State is the only record of what exists: never delete it with the stack.
  lifecycle { prevent_destroy = true }
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration { status = "Enabled" }
}

# A key this account owns (rotated yearly), so access to state can be
# audited and revoked through the key policy, not only the bucket's.
resource "aws_kms_key" "state" {
  description             = "MedAgent Terraform state"
  enable_key_rotation     = true
  deletion_window_in_days = 30
  lifecycle { prevent_destroy = true }
}

resource "aws_kms_alias" "state" {
  name          = "alias/medagent-tfstate"
  target_key_id = aws_kms_key.state.key_id
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.state.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "state" {
  bucket = aws_s3_bucket.state.id
  rule { object_ownership = "BucketOwnerEnforced" }
}

# TLS only.
resource "aws_s3_bucket_policy" "state" {
  bucket = aws_s3_bucket.state.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.state.arn, "${aws_s3_bucket.state.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

# Old state versions are kept 90 days, enough to roll back a bad apply.
resource "aws_s3_bucket_lifecycle_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration { noncurrent_days = 90 }
  }
}

output "bucket" {
  description = "Pass as TF_STATE_BUCKET to make tf-init."
  value       = aws_s3_bucket.state.bucket
}
