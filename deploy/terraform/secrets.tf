# The app's Kubernetes Secret (medagent-secrets, keys as in
# deploy/k8s/base/secret.example.yaml) is synced from this Secrets Manager
# secret by the External Secrets Operator (addons.tf). Nothing secret is kept
# in Terraform state: the passwords are ephemeral and the secret's value is a
# write-only argument, sent only when secrets_version changes.

ephemeral "random_password" "jwt" {
  length  = 64
  special = false
}

ephemeral "random_password" "admin" {
  length  = 24
  special = false
}

resource "aws_secretsmanager_secret" "app" {
  name                    = "medagent/${var.env}"
  description             = "MedAgent app secrets (synced to the medagent-secrets Kubernetes Secret)"
  recovery_window_in_days = var.deletion_protection ? 30 : 0
}

resource "aws_secretsmanager_secret_version" "app" {
  secret_id = aws_secretsmanager_secret.app.id
  secret_string_wo = jsonencode({
    POSTGRES_DSN             = "postgresql://${module.db.db_instance_username}:${ephemeral.random_password.db.result}@${module.db.db_instance_address}:5432/medagent?sslmode=require"
    JWT_SECRET_KEY           = ephemeral.random_password.jwt.result
    ADMIN_BOOTSTRAP_USERNAME = "admin"
    ADMIN_BOOTSTRAP_PASSWORD = ephemeral.random_password.admin.result
    GROQ_API_KEY             = var.groq_api_key
    LLM_API_KEY              = var.llm_api_key
  })
  secret_string_wo_version = var.secrets_version
}
