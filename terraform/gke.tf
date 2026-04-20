# GKE Cluster for LLM Router
resource "google_container_cluster" "llm_router" {
  name     = "${var.project_name}-cluster"
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
  }

  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
}

resource "google_container_node_pool" "router_nodes" {
  name       = "router-pool"
  cluster    = google_container_cluster.llm_router.name
  location   = var.region
  node_count = 2

  node_config {
    preemptible  = true
    machine_type = "e2-standard-4"

    service_account = google_service_account.gke_sa.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      env = var.environment
      app = "llm-router"
    }

    tags = ["llm-router"]
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 5
  }
}

resource "google_service_account" "gke_sa" {
  account_id   = "${var.project_name}-gke"
  display_name = "GKE Service Account"
}

resource "google_project_iam_member" "gke_sa_roles" {
  for_each = toset([
    "roles/container.nodeServiceAccount",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_sa.email}"
}