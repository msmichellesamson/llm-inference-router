terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.5"
}

variable "project_id" {
  description = "GCP project ID for LLM inference router deployment"
  type        = string
}

variable "region" {
  description = "GCP region for resource deployment"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for compute instances"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "redis_memory_size_gb" {
  description = "Redis instance memory size in GB"
  type        = number
  default     = 1
}

variable "gke_node_count" {
  description = "Number of GKE nodes per zone"
  type        = number
  default     = 2
}

variable "gke_machine_type" {
  description = "GKE node machine type"
  type        = string
  default     = "e2-standard-4"
}

locals {
  name_prefix = "llm-router-${var.environment}"
  common_labels = {
    project     = "llm-inference-router"
    environment = var.environment
    managed-by  = "terraform"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "redis.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudtrace.googleapis.com",
    "servicenetworking.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
  
  disable_dependent_services = true
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "${local.name_prefix}-vpc"
  auto_create_subnetworks = false
  
  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "${local.name_prefix}-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# Cloud NAT for outbound internet access
resource "google_compute_router" "router" {
  name    = "${local.name_prefix}-router"
  region  = var.region
  network = google_compute_network.vpc.id
}

resource "google_compute_router_nat" "nat" {
  name                               = "${local.name_prefix}-nat"
  router                            = google_compute_router.router.name
  region                            = var.region
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# Redis Instance for caching
resource "google_compute_global_address" "private_ip_address" {
  name          = "${local.name_prefix}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

resource "google_redis_instance" "cache" {
  name           = "${local.name_prefix}-cache"
  tier           = "STANDARD_HA"
  memory_size_gb = var.redis_memory_size_gb
  region         = var.region
  
  authorized_network = google_compute_network.vpc.id
  redis_version      = "REDIS_7_0"
  display_name       = "LLM Router Cache"
  
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    notify-keyspace-events = "Ex"
  }
  
  labels = local.common_labels
  
  depends_on = [
    google_service_networking_connection.private_vpc_connection,
    google_project_service.apis
  ]
}

# GKE Cluster
resource "google_container_cluster" "cluster" {
  name     = "${local.name_prefix}-cluster"
  location = var.region
  
  remove_default_node_pool = true
  initial_node_count       = 1
  
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name
  
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  
  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
      "APISERVER",
      "CONTROLLER_MANAGER",
      "SCHEDULER"
    ]
    
    managed_prometheus {
      enabled = true
    }
  }
  
  logging_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
      "APISERVER",
      "CONTROLLER_MANAGER",
      "SCHEDULER"
    ]
  }
  
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  addons_config {
    http_load_balancing {
      disabled = false
    }
    
    horizontal_pod_autoscaling {
      disabled = false
    }
    
    network_policy_config {
      disabled = false
    }
  }
  
  network_policy {
    enabled = true
  }
  
  resource_labels = local.common_labels
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "${local.name_prefix}-nodes"
  location   = var.region
  cluster    = google_container_cluster.cluster.name
  node_count = var.gke_node_count
  
  node_config {
    preemptible  = var.environment != "prod"
    machine_type = var.gke_machine_type
    disk_size_gb = 50
    disk_type    = "pd-ssd"
    
    service_account = google_service_account.gke_sa.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    labels = local.common_labels
    
    tags = ["llm-router", var.environment]
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  autoscaling {
    min_node_count = 1
    max_node_count = var.environment == "prod" ? 10 : 5
  }
}

# Service Account for GKE
resource "google_service_account" "gke_sa" {
  account_id   = "${local.name_prefix}-gke-sa"
  display_name = "LLM Router GKE Service Account"
}

resource "google_project_iam_member" "gke_sa_roles" {
  for_each = toset([
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter",
    "roles/cloudtrace.agent",
    "roles/redis.editor"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_sa.email}"
}

# Monitoring Dashboard
resource "google_monitoring_dashboard" "llm_router" {
  dashboard_json = jsonencode({
    displayName = "LLM Inference Router - ${title(var.environment)}"
    mosaicLayout = {
      tiles = [
        {
          width = 6
          height = 4
          widget = {
            title = "Request Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"k8s_container\" resource.labels.container_name=\"llm-router\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_RATE"
                    }
                  }
                }
              }]
              yAxis = {
                label = "Requests/sec"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          width = 6
          height = 4
          xPos = 6
          widget = {
            title = "Response Latency"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"k8s_container\" resource.labels.container_name=\"llm-router\" metric.type=\"custom.googleapis.com/llm_router/response_latency\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
              }]
              yAxis = {
                label = "Latency (ms)"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          width = 6
          height = 4
          yPos = 4
          widget = {
            title = "Model Usage Distribution"
            pieChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"k8s_container\" metric.type=\"custom.googleapis.com/llm_router/model_usage\""
                    aggregation = {
                      alignmentPeriod = "300s"
                      perSeriesAligner = "ALIGN_SUM"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields = ["metric.label.model_name"]
                    }
                  }
                }
              }]
            }
          }
        },
        {
          width = 6
          height = 4
          xPos = 6
          yPos = 4
          widget = {
            title = "Cache Hit Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"k8s_container\" metric.type=\"custom.googleapis.com/llm_router/cache_hit_rate\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
              }]
              yAxis = {
                label = "Hit Rate (%)"
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })
}

# Alerting Policies
resource "google_monitoring_alert_policy" "high_latency" {
  display_name = "LLM Router High Latency - ${title(var.environment)}"
  combiner     = "OR"
  
  conditions {
    display_name = "High response latency"
    
    condition_threshold {
      filter         = "resource.type=\"k8s_container\" resource.labels.container_name=\"llm-router\" metric.type=\"custom.googleapis.com/llm_router/response_latency\""
      duration       = "300s"
      comparison     = "COMPARISON_GREATER_THAN"
      threshold_value = var.environment == "prod" ? 2000 : 5000
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = []
  
  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "low_cache_hit_rate" {
  display_name = "LLM Router Low Cache Hit Rate - ${title(var.environment)}"
  combiner     = "OR"
  
  conditions {
    display_name = "Cache hit rate below threshold"
    
    condition_threshold {
      filter         = "resource.type=\"k8s_container\" metric.type=\"custom.googleapis.com/llm_router/cache_hit_rate\""
      duration       = "600s"
      comparison     = "COMPARISON_LESS_THAN"
      threshold_value = 0.7
      
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = []
  
  alert_strategy {
    auto_close = "1800s"
  }
}

# Outputs
output "gke_cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.cluster.name
}

output "gke_cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.cluster.endpoint
  sensitive   = true
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.cache.host
}

output "redis_port" {
  description = "Redis instance port"
  value       = google_redis_instance.cache.port
}

output "vpc_network" {
  description = "VPC network name"
  value       = google_compute_network.vpc.name
}

output "service_account_email" {
  description = "GKE service account email"
  value       = google_service_account.gke_sa.email
}