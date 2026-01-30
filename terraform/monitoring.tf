# Monitoring Infrastructure for LLM Inference Router
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

# Variables
variable "project_id" {
  description = "GCP project ID for monitoring resources"
  type        = string
}

variable "cluster_name" {
  description = "GKE cluster name where monitoring will be deployed"
  type        = string
}

variable "cluster_location" {
  description = "GKE cluster location"
  type        = string
}

variable "monitoring_namespace" {
  description = "Kubernetes namespace for monitoring stack"
  type        = string
  default     = "monitoring"
}

variable "grafana_admin_password" {
  description = "Admin password for Grafana"
  type        = string
  sensitive   = true
}

variable "prometheus_retention" {
  description = "Prometheus data retention period"
  type        = string
  default     = "30d"
}

variable "alertmanager_slack_webhook" {
  description = "Slack webhook URL for alertmanager notifications"
  type        = string
  sensitive   = true
  default     = ""
}

# Data sources
data "google_client_config" "default" {}

data "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.cluster_location
}

# Kubernetes provider configuration
provider "kubernetes" {
  host  = "https://${data.google_container_cluster.primary.endpoint}"
  token = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(
    data.google_container_cluster.primary.master_auth[0].cluster_ca_certificate,
  )
}

provider "helm" {
  kubernetes {
    host  = "https://${data.google_container_cluster.primary.endpoint}"
    token = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(
      data.google_container_cluster.primary.master_auth[0].cluster_ca_certificate,
    )
  }
}

# Monitoring namespace
resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = var.monitoring_namespace
    labels = {
      name        = var.monitoring_namespace
      purpose     = "monitoring"
      environment = "production"
    }
  }
}

# Storage class for monitoring persistent volumes
resource "kubernetes_storage_class" "monitoring_ssd" {
  metadata {
    name = "monitoring-ssd"
  }
  storage_provisioner    = "kubernetes.io/gce-pd"
  reclaim_policy        = "Retain"
  volume_binding_mode   = "WaitForFirstConsumer"
  allow_volume_expansion = true
  parameters = {
    type               = "pd-ssd"
    replication-type   = "regional-pd"
    zones              = "${var.cluster_location}-a,${var.cluster_location}-b"
  }
}

# Prometheus Operator via Helm
resource "helm_release" "prometheus_operator" {
  name       = "prometheus-operator"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "55.5.0"
  namespace  = kubernetes_namespace.monitoring.metadata[0].name

  values = [
    yamlencode({
      nameOverride     = "prometheus"
      fullnameOverride = "prometheus"

      # Prometheus configuration
      prometheus = {
        prometheusSpec = {
          retention                   = var.prometheus_retention
          retentionSize              = "50GB"
          serviceMonitorSelectorNilUsesHelmValues = false
          podMonitorSelectorNilUsesHelmValues     = false
          ruleSelectorNilUsesHelmValues           = false
          storageSpec = {
            volumeClaimTemplate = {
              spec = {
                storageClassName = kubernetes_storage_class.monitoring_ssd.metadata[0].name
                accessModes      = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "100Gi"
                  }
                }
              }
            }
          }
          resources = {
            requests = {
              cpu    = "1000m"
              memory = "2Gi"
            }
            limits = {
              cpu    = "2000m"
              memory = "4Gi"
            }
          }
          additionalScrapeConfigs = [
            {
              job_name        = "llm-router-app"
              static_configs = [{
                targets = ["llm-router-service.default.svc.cluster.local:8080"]
              }]
              scrape_interval = "30s"
              metrics_path    = "/metrics"
            }
          ]
        }
        service = {
          type = "ClusterIP"
        }
        ingress = {
          enabled = false
        }
      }

      # Grafana configuration
      grafana = {
        enabled = true
        adminPassword = var.grafana_admin_password
        persistence = {
          enabled          = true
          size            = "10Gi"
          storageClassName = kubernetes_storage_class.monitoring_ssd.metadata[0].name
        }
        resources = {
          requests = {
            cpu    = "100m"
            memory = "256Mi"
          }
          limits = {
            cpu    = "500m"
            memory = "1Gi"
          }
        }
        service = {
          type = "LoadBalancer"
          annotations = {
            "cloud.google.com/load-balancer-type" = "External"
          }
        }
        grafana.ini = {
          server = {
            root_url = "http://grafana.${var.project_id}.com"
          }
          auth = {
            disable_login_form = false
          }
          "auth.anonymous" = {
            enabled = false
          }
        }
        datasources = {
          "datasources.yaml" = {
            apiVersion = 1
            datasources = [
              {
                name      = "Prometheus"
                type      = "prometheus"
                url       = "http://prometheus-prometheus-prometheus:9090"
                access    = "proxy"
                isDefault = true
              }
            ]
          }
        }
        dashboardProviders = {
          "dashboardproviders.yaml" = {
            apiVersion = 1
            providers = [
              {
                name            = "default"
                orgId           = 1
                folder          = ""
                type            = "file"
                disableDeletion = false
                editable        = true
                options = {
                  path = "/var/lib/grafana/dashboards/default"
                }
              }
            ]
          }
        }
        dashboards = {
          default = {
            llm-router-dashboard = {
              gnetId    = null
              revision  = null
              datasource = "Prometheus"
              json = jsonencode({
                dashboard = {
                  id    = null
                  title = "LLM Router Dashboard"
                  tags  = ["llm", "router", "inference"]
                  timezone = "browser"
                  panels = [
                    {
                      id    = 1
                      title = "Request Rate"
                      type  = "stat"
                      targets = [
                        {
                          expr    = "rate(http_requests_total{job=\"llm-router-app\"}[5m])"
                          refId   = "A"
                          legendFormat = "Requests/sec"
                        }
                      ]
                      gridPos = {
                        h = 8
                        w = 12
                        x = 0
                        y = 0
                      }
                    },
                    {
                      id    = 2
                      title = "Response Latency"
                      type  = "graph"
                      targets = [
                        {
                          expr    = "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"llm-router-app\"}[5m]))"
                          refId   = "A"
                          legendFormat = "95th percentile"
                        },
                        {
                          expr    = "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{job=\"llm-router-app\"}[5m]))"
                          refId   = "B"
                          legendFormat = "50th percentile"
                        }
                      ]
                      gridPos = {
                        h = 8
                        w = 12
                        x = 12
                        y = 0
                      }
                    },
                    {
                      id    = 3
                      title = "Model Usage Distribution"
                      type  = "piechart"
                      targets = [
                        {
                          expr    = "sum by (model_name) (rate(llm_router_requests_total[5m]))"
                          refId   = "A"
                          legendFormat = "{{model_name}}"
                        }
                      ]
                      gridPos = {
                        h = 8
                        w = 12
                        x = 0
                        y = 8
                      }
                    },
                    {
                      id    = 4
                      title = "Cache Hit Rate"
                      type  = "stat"
                      targets = [
                        {
                          expr    = "rate(llm_router_cache_hits_total[5m]) / (rate(llm_router_cache_hits_total[5m]) + rate(llm_router_cache_misses_total[5m])) * 100"
                          refId   = "A"
                          legendFormat = "Hit Rate %"
                        }
                      ]
                      gridPos = {
                        h = 8
                        w = 12
                        x = 12
                        y = 8
                      }
                    },
                    {
                      id    = 5
                      title = "Error Rate"
                      type  = "graph"
                      targets = [
                        {
                          expr    = "rate(http_requests_total{job=\"llm-router-app\",status=~\"5..\"}[5m])"
                          refId   = "A"
                          legendFormat = "5xx Errors"
                        },
                        {
                          expr    = "rate(http_requests_total{job=\"llm-router-app\",status=~\"4..\"}[5m])"
                          refId   = "B"
                          legendFormat = "4xx Errors"
                        }
                      ]
                      gridPos = {
                        h = 8
                        w = 24
                        x = 0
                        y = 16
                      }
                    }
                  ]
                  time = {
                    from = "now-1h"
                    to   = "now"
                  }
                  refresh = "5s"
                }
              })
            }
          }
        }
      }

      # Alertmanager configuration
      alertmanager = {
        alertmanagerSpec = {
          storage = {
            volumeClaimTemplate = {
              spec = {
                storageClassName = kubernetes_storage_class.monitoring_ssd.metadata[0].name
                accessModes      = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "10Gi"
                  }
                }
              }
            }
          }
          resources = {
            requests = {
              cpu    = "100m"
              memory = "128Mi"
            }
            limits = {
              cpu    = "500m"
              memory = "512Mi"
            }
          }
        }
        config = var.alertmanager_slack_webhook != "" ? {
          global = {
            slack_api_url = var.alertmanager_slack_webhook
          }
          route = {
            group_by        = ["alertname"]
            group_wait      = "10s"
            group_interval  = "10s"
            repeat_interval = "1h"
            receiver        = "web.hook"
          }
          receivers = [
            {
              name = "web.hook"
              slack_configs = [
                {
                  channel    = "#alerts"
                  title      = "Alert - {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
                  text       = "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                  send_resolved = true
                }
              ]
            }
          ]
        } : {}
      }

      # Node Exporter
      nodeExporter = {
        enabled = true
      }

      # Kube State Metrics
      kubeStateMetrics = {
        enabled = true
      }

      # Default rules
      defaultRules = {
        create = true
        rules = {
          alertmanager              = true
          etcd                     = true
          configReloaders          = true
          general                  = true
          k8s                      = true
          kubeApiserverAvailability = true
          kubeApiserverBurnrate    = true
          kubeApiserverHistogram   = true
          kubeApiserverSlos        = true
          kubelet                  = true
          kubeProxy                = true
          kubePrometheusGeneral    = true
          kubePrometheusNodeRecording = true
          kubernetesApps           = true
          kubernetesResources      = true
          kubernetesStorage        = true
          kubernetesSystem         = true
          kubeScheduler            = true
          kubeStateMetrics         = true
          network                  = true
          node                     = true
          nodeExporterAlerting     = true
          nodeExporterRecording    = true
          prometheus               = true
          prometheusOperator       = true
        }
      }
    })
  ]

  depends_on = [kubernetes_namespace.monitoring, kubernetes_storage_class.monitoring_ssd]
}

# Custom PrometheusRule for LLM Router specific alerts
resource "kubernetes_manifest" "llm_router_alerts" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "PrometheusRule"
    metadata = {
      name      = "llm-router-alerts"
      namespace = var.monitoring_namespace
      labels = {
        prometheus = "prometheus-prometheus-prometheus"
        role       = "alert-rules"
      }
    }
    spec = {
      groups = [
        {
          name = "llm-router.rules"
          rules = [
            {
              alert = "LLMRouterHighErrorRate"
              expr  = "rate(http_requests_total{job=\"llm-router-app\",status=~\"5..\"}[5m]) > 0.1"
              for   = "5m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "LLM Router high error rate"
                description = "LLM Router is experiencing {{ $value }} errors per second"
              }
            },
            {
              alert = "LLMRouterHighLatency"
              expr  = "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"llm-router-app\"}[5m])) > 5"
              for   = "10m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "LLM Router high latency"
                description = "95th percentile latency is {{ $value }}s"
              }
            },
            {
              alert = "LLMRouterLowCacheHitRate"
              expr  = "rate(llm_router_cache_hits_total[5m]) / (rate(llm_router_cache_hits_total[5m]) + rate(llm_router_cache_misses_total[5m])) < 0.3"
              for   = "15m"
              labels = {
                severity = "info"
              }
              annotations = {
                summary     = "LLM Router low cache hit rate"
                description = "Cache hit rate is {{ $value | humanizePercentage }}"
              }
            },
            {
              alert = "LLMRouterModelUnavailable"
              expr  = "up{job=\"llm-router-app\"} == 0"
              for   = "1m"
              labels = {
                severity = "critical"
              }
              annotations = {
                summary     = "LLM Router is down"
                description = "LLM Router has been down for more than 1 minute"
              }
            }
          ]
        }
      ]
    }
  }

  depends_on = [helm_release.prometheus_operator]
}

# ServiceMonitor for the LLM Router application
resource "kubernetes_manifest" "llm_router_service_monitor" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "llm-router-monitor"
      namespace = var.monitoring_namespace
      labels = {
        app                           = "llm-router"
        release                      = "prometheus-operator"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          app = "llm-router"
        }
      }
      namespaceSelector = {
        matchNames = ["default"]
      }
      endpoints = [
        {
          port     = "http"
          path     = "/metrics"
          interval = "30s"
        }
      ]
    }
  }

  depends_on = [helm_release.prometheus_operator]
}

# Outputs
output "grafana_admin_password" {
  description = "Admin password for Grafana (retrieve with: terraform output -raw grafana_admin_password)"
  value       = var.grafana_admin_password
  sensitive   = true
}

output "prometheus_url" {
  description = "Internal Prometheus URL"
  value       = "http://prometheus-prometheus-prometheus.${var.monitoring_namespace}.svc.cluster.local:9090"
}

output "grafana_service_name" {
  description = "Grafana service name for port forwarding"
  value       = "prometheus-operator-grafana"
}

output "monitoring_namespace" {
  description = "Kubernetes namespace where monitoring stack is deployed"
  value       = var.monitoring_namespace
}