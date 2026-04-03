# Redis cluster for caching and session management
resource "google_redis_instance" "llm_cache" {
  name           = "${var.project_name}-cache"
  tier           = "STANDARD_HA"
  memory_size_gb = 4
  region         = var.region
  
  redis_version     = "REDIS_7_0"
  display_name      = "LLM Router Cache"
  
  auth_enabled               = true
  transit_encryption_mode    = "SERVER_AUTH"
  connect_mode              = "PRIVATE_SERVICE_ACCESS"
  
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    timeout          = "300"
  }
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 2
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }
  
  labels = {
    environment = var.environment
    component   = "cache"
    managed_by  = "terraform"
  }
}

resource "google_redis_instance" "llm_sessions" {
  name           = "${var.project_name}-sessions"
  tier           = "BASIC"
  memory_size_gb = 1
  region         = var.region
  
  redis_version  = "REDIS_7_0"
  display_name   = "LLM Router Sessions"
  
  auth_enabled            = true
  transit_encryption_mode = "SERVER_AUTH"
  connect_mode           = "PRIVATE_SERVICE_ACCESS"
  
  labels = {
    environment = var.environment
    component   = "sessions"
    managed_by  = "terraform"
  }
}

output "redis_cache_host" {
  description = "Redis cache instance host"
  value       = google_redis_instance.llm_cache.host
  sensitive   = true
}

output "redis_cache_port" {
  description = "Redis cache instance port"
  value       = google_redis_instance.llm_cache.port
}

output "redis_sessions_host" {
  description = "Redis sessions instance host"
  value       = google_redis_instance.llm_sessions.host
  sensitive   = true
}