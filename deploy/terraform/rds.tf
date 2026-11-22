# Postgres 16 with pgvector available (the first migration runs CREATE
# EXTENSION vector as the master user, which has rds_superuser).

resource "aws_security_group" "db" {
  name        = "${local.name}-db"
  description = "Postgres, reachable only from the EKS nodes (pods use the node SG)"
  vpc_id      = module.vpc.vpc_id
}

resource "aws_vpc_security_group_ingress_rule" "db_from_nodes" {
  security_group_id            = aws_security_group.db.id
  referenced_security_group_id = module.eks.node_security_group_id
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
  description                  = "Postgres from EKS nodes"
}

# Never stored in state: generated per run, written to RDS and to the secret
# only when secrets_version changes (write-only arguments).
ephemeral "random_password" "db" {
  length  = 32
  special = false
}

module "db" {
  source  = "terraform-aws-modules/rds/aws"
  version = "7.2.2"

  identifier = local.name

  engine               = "postgres"
  engine_version       = "16"
  family               = "postgres16"
  major_engine_version = "16"
  instance_class       = var.db_instance_class

  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_allocated_storage * 5
  storage_encrypted     = true

  db_name  = "medagent"
  username = "medagent"
  port     = 5432

  manage_master_user_password = false
  password_wo                 = ephemeral.random_password.db.result
  password_wo_version         = var.secrets_version

  multi_az               = var.db_multi_az
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false

  backup_retention_period = var.db_backup_retention_days
  backup_window           = "03:00-04:00"
  maintenance_window      = "Sun:04:30-Sun:05:30"
  copy_tags_to_snapshot   = true

  deletion_protection              = var.deletion_protection
  skip_final_snapshot              = !var.deletion_protection
  final_snapshot_identifier_prefix = "${local.name}-final"

  performance_insights_enabled = true

  # The app connects with sslmode=require; refuse anything else.
  parameters = [
    { name = "rds.force_ssl", value = "1" },
  ]
}
