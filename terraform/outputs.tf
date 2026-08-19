output "orchestrator_url" {
  description = "The URL of the deployed orchestrator service."
  value       = google_cloud_run_v2_service.orchestrator.uri
}

output "service_urls" {
  description = "The URLs of all deployed services."
  value = { for k, v in merge(
    google_cloud_run_v2_service.tool_services,
    google_cloud_run_v2_service.specialist_agents,
    { "orchestrator" = google_cloud_run_v2_service.orchestrator }
  ) : k => v.uri }
}