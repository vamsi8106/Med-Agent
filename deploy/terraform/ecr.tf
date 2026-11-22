locals {
  images = ["medagent", "medical-mcp-bridge", "healthcare-mcp", "research-mcp"]
}

resource "aws_ecr_repository" "image" {
  for_each = toset(local.images)

  name = each.value
  # A tag always means the same image: a rollback deploys exactly what ran before.
  image_tag_mutability = "IMMUTABLE"
  force_delete         = !var.deletion_protection

  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "KMS" }
}

resource "aws_ecr_lifecycle_policy" "image" {
  for_each   = aws_ecr_repository.image
  repository = each.value.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 20 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 20 }
      action       = { type = "expire" }
    }]
  })
}
