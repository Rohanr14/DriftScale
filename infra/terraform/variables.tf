variable "project_name" {
  description = "Short name used as a prefix for demo resources."
  type        = string
  default     = "driftscale-demo"
}

variable "aws_region" {
  description = "AWS region for the ephemeral ECS Fargate demo."
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR block for the single demo VPC."
  type        = string
  default     = "10.42.0.0/16"
}

variable "allowed_http_cidr" {
  description = "CIDR allowed to reach the public ALB."
  type        = string
  default     = "0.0.0.0/0"
}

variable "container_port" {
  description = "FastAPI container port."
  type        = number
  default     = 8000
}

variable "container_image" {
  description = "Optional prebuilt container image URI. Defaults to the generated ECR repo latest tag."
  type        = string
  default     = ""
}

variable "initial_desired_count" {
  description = "Initial ECS desired count for the demo service."
  type        = number
  default     = 1
}

variable "budget_email" {
  description = "Optional email subscriber for the $10 budget alert."
  type        = string
  default     = ""
}
