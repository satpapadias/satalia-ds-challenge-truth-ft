terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.35"
    }
  }
  backend "gcs" {
    bucket = "x-wppai-researchlab-wpptestbed-truthclf-tfstate"
    prefix = "cloud-run"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "cloudbuild.googleapis.com",
    "aiplatform.googleapis.com",
  ])
  service                    = each.key
  disable_dependent_services = true
  disable_on_destroy         = false
}

resource "google_artifact_registry_repository" "repo" {
  depends_on    = [google_project_service.apis]
  provider      = google-beta
  location      = var.region
  repository_id = var.repository
  description   = "Container images for the truthclf agent network."
  format        = "DOCKER"
}

locals {
  service_names = [
    "data-tools",
    "model-tools",
    "zero-shot-predictor",
    "fine-tuned-predictor",
    "explainer",
    "orchestrator",
  ]
  service_account_ids = {
    "data-tools"           = "sa-truthclf-data-tools"
    "model-tools"          = "sa-truthclf-model-tools"
    "zero-shot-predictor"  = "sa-truthclf-zs-pred"
    "fine-tuned-predictor" = "sa-truthclf-ft-pred"
    "explainer"            = "sa-truthclf-explainer"
    "orchestrator"         = "sa-truthclf-orchestrator"
  }
}

resource "google_service_account" "service_accounts" {
  for_each     = toset(local.service_names)
  account_id   = local.service_account_ids[each.key]
  display_name = "SA for truthclf ${each.key} service"
}

resource "google_secret_manager_secret" "orchestrator_token" {
  depends_on = [google_project_service.apis]
  secret_id  = "truthclf-orchestrator-token"
  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
}

resource "google_secret_manager_secret_version" "orchestrator_token_version" {
  secret      = google_secret_manager_secret.orchestrator_token.id
  secret_data = var.orchestrator_token
}

resource "google_secret_manager_secret_iam_member" "orchestrator_token_accessor" {
  secret_id = google_secret_manager_secret.orchestrator_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.service_accounts["orchestrator"].email}"
}

locals {
  tools_image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository}/truthclf-tools:latest"
  agent_image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository}/truthclf-agent:latest"
  agent_env = {
    HOST                  = "0.0.0.0"
    GOOGLE_CLOUD_PROJECT  = var.project_id
    GOOGLE_CLOUD_LOCATION = var.region
  }
}

resource "google_cloud_run_v2_service" "tool_services" {
  depends_on = [google_project_service.apis]
  for_each   = {
    "data-tools" = {
      image   = local.tools_image
      port    = 8080
      command = ["python", "-m", "truthclf_mcp.data_tools", "--host", "0.0.0.0"]
    },
    "model-tools" = {
      image   = local.tools_image
      port    = 8080
      command = ["python", "-m", "truthclf_mcp.model_tools", "--host", "0.0.0.0"]
    }
  }
  name                = "truthclf-${each.key}"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.service_accounts[each.key].email
    containers {
      image   = each.value.image
      command = each.value.command
      ports { container_port = each.value.port }
      startup_probe {
        tcp_socket { port = each.value.port }
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 12
      }
      dynamic "env" {
        for_each = local.agent_env
        content {
          name  = env.key
          value = env.value
        }
      }
    }
  }
}

resource "google_cloud_run_v2_service" "specialist_agents" {
  depends_on = [google_cloud_run_v2_service.tool_services]
  for_each   = {
    "zero-shot-predictor" = {
      image   = local.agent_image
      port    = 8080
      command = ["python", "-m", "truthclf_agents.zero_shot", "--host", "0.0.0.0"]
      memory  = "8Gi"
      cpu     = "2"
      env = { MODEL_TOOLS_URL = "${google_cloud_run_v2_service.tool_services["model-tools"].uri}/mcp" }
    },
    "fine-tuned-predictor" = {
      image   = local.agent_image
      port    = 8080
      command = ["python", "-m", "truthclf_agents.fine_tuned", "--host", "0.0.0.0"]
      memory  = "8Gi"
      cpu     = "2"
      env = { MODEL_TOOLS_URL = "${google_cloud_run_v2_service.tool_services["model-tools"].uri}/mcp" }
    },
    "explainer" = {
      image   = local.agent_image
      port    = 8080
      command = ["python", "-m", "truthclf_agents.explainer", "--host", "0.0.0.0"]
      memory  = "4Gi"
      cpu     = "2"
      env = { MODEL_TOOLS_URL = "${google_cloud_run_v2_service.tool_services["model-tools"].uri}/mcp" }
    }
  }

  name                = "truthclf-${each.key}"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.service_accounts[each.key].email
    containers {
      image   = each.value.image
      command = each.value.command
      ports { container_port = each.value.port }
      
      resources {
        limits = {
          memory = each.value.memory
          cpu    = each.value.cpu
        }
      }

      startup_probe {
        tcp_socket { port = each.value.port }
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 24
      }
      dynamic "env" {
        for_each = merge(local.agent_env, each.value.env)
        content {
          name  = env.key
          value = env.value
        }
      }
    }
  }
}

resource "google_cloud_run_v2_service" "orchestrator" {
  depends_on = [google_cloud_run_v2_service.specialist_agents]
  name                = "truthclf-orchestrator"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.service_accounts["orchestrator"].email
    containers {
      image   = local.agent_image
      command = ["python", "-m", "truthclf_agents.orchestrator", "--host", "0.0.0.0"]
      ports { container_port = 8080 }
      
      resources {
        limits = {
          memory = "2Gi"
          cpu    = "1"
        }
      }

      startup_probe {
        tcp_socket { port = 8080 }
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 12
      }
      dynamic "env" {
        for_each = merge(local.agent_env, {
          ZERO_SHOT_AGENT_URL  = google_cloud_run_v2_service.specialist_agents["zero-shot-predictor"].uri
          FINE_TUNED_AGENT_URL = google_cloud_run_v2_service.specialist_agents["fine-tuned-predictor"].uri
          EXPLAINER_AGENT_URL  = google_cloud_run_v2_service.specialist_agents["explainer"].uri
          DATA_TOOLS_URL       = "${google_cloud_run_v2_service.tool_services["data-tools"].uri}/mcp"
          POOL_WEIGHT          = var.pool_weight
          MAX_POINTS           = var.max_points
        })
        content {
          name  = env.key
          value = env.value
        }
      }
      dynamic "env" {
        for_each = [1]
        content {
          name = "ORCHESTRATOR_TOKEN"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.orchestrator_token.secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }
}

locals {
  call_graph = {
    orchestrator         = ["zero-shot-predictor", "fine-tuned-predictor", "explainer", "data-tools"],
    zero-shot-predictor  = ["model-tools"],
    fine-tuned-predictor = ["model-tools"],
    explainer            = ["model-tools"],
  }

  all_services = merge(
    google_cloud_run_v2_service.tool_services,
    google_cloud_run_v2_service.specialist_agents,
    { "orchestrator" = google_cloud_run_v2_service.orchestrator },
  )
}

resource "google_cloud_run_v2_service_iam_member" "invoker_bindings" {
  for_each = { for pair in setproduct(local.service_names, local.service_names) :
    "${pair[0]}-to-${pair[1]}" => { caller = pair[0], callee = pair[1] } if contains(lookup(local.call_graph, pair[0], []), pair[1])
  }
  project  = var.project_id
  location = var.region
  name     = local.all_services[each.value.callee].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.service_accounts[each.value.caller].email}"
}
