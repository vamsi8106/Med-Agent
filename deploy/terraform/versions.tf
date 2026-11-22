terraform {
  # 1.11: write-only arguments, used so no password is ever stored in state.
  required_version = ">= 1.11"

  required_providers {
    aws        = { source = "hashicorp/aws", version = "= 6.73.0" }
    helm       = { source = "hashicorp/helm", version = "= 3.3.0" }
    kubernetes = { source = "hashicorp/kubernetes", version = "= 3.5.0" }
    random     = { source = "hashicorp/random", version = "= 3.9.1" }
  }

  # Partial configuration: `make tf-init` passes the bucket (from ./bootstrap),
  # a key per environment, the region, encryption and S3-native locking.
  backend "s3" {}
}
