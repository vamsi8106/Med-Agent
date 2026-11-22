resource "aws_acm_certificate" "app" {
  domain_name       = var.domain_name
  validation_method = "DNS"
  lifecycle { create_before_destroy = true }
}

locals {
  validation_records = {
    for o in aws_acm_certificate.app.domain_validation_options : o.domain_name => {
      name  = o.resource_record_name
      type  = o.resource_record_type
      value = o.resource_record_value
    }
  }
}

# With a Route 53 zone, validation is automatic; without one, add the records
# from the acm_validation_records output at your DNS provider (the ALB fails to
# get HTTPS until the certificate is issued).
resource "aws_route53_record" "validation" {
  for_each = var.route53_zone_id != "" ? local.validation_records : {}

  zone_id         = var.route53_zone_id
  name            = each.value.name
  type            = each.value.type
  records         = [each.value.value]
  ttl             = 300
  allow_overwrite = true
}

resource "aws_acm_certificate_validation" "app" {
  count                   = var.route53_zone_id != "" ? 1 : 0
  certificate_arn         = aws_acm_certificate.app.arn
  validation_record_fqdns = [for r in aws_route53_record.validation : r.fqdn]
}
