variable "project_id" {
  description = "The GCP project ID to deploy into."
  type        = string
}

variable "region" {
  description = "The GCP region to deploy services into."
  type        = string
  default     = "us-central1"
}

variable "repository" {
  description = "The name of the Artifact Registry repository for the images."
  type        = string
  default     = "truthclf-images"
}

variable "orchestrator_token" {
  description = "The public bearer token for the orchestrator's /verify endpoint. Will be stored in Secret Manager."
  type        = string
  sensitive   = true
}

variable "pool_weight" {
  description = "The weight for log-odds pooling in the orchestrator."
  type        = number
  default     = 1.0
}

variable "max_points" {
  description = "The maximum number of points the orchestrator will accept in a batch."
  type        = number
  default     = 50
}