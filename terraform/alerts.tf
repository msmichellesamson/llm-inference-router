# Prometheus alerting rules for LLM router
resource "kubernetes_config_map" "prometheus_alerts" {
  metadata {
    name      = "llm-router-alerts"
    namespace = "monitoring"
    labels = {
      app = "llm-inference-router"
    }
  }

  data = {
    "alerts.yml" = yamlencode({
      groups = [{
        name = "llm-router"
        rules = [
          {
            alert = "HighLatency"
            expr  = "histogram_quantile(0.95, llm_request_duration_seconds) > 10"
            for   = "2m"
            labels = {
              severity = "warning"
            }
            annotations = {
              summary     = "High request latency detected"
              description = "95th percentile latency is {{ $value }}s"
            }
          },
          {
            alert = "CircuitBreakerOpen"
            expr  = "llm_circuit_breaker_state == 1"
            for   = "1m"
            labels = {
              severity = "critical"
            }
            annotations = {
              summary     = "Circuit breaker is open"
              description = "Model {{ $labels.model }} circuit breaker is open"
            }
          },
          {
            alert = "HighErrorRate"
            expr  = "rate(llm_requests_failed_total[5m]) / rate(llm_requests_total[5m]) > 0.1"
            for   = "3m"
            labels = {
              severity = "warning"
            }
            annotations = {
              summary     = "High error rate detected"
              description = "Error rate is {{ $value | humanizePercentage }}"
            }
          }
        ]
      }]
    })
  }
}

# ServiceMonitor for Prometheus to scrape metrics
resource "kubernetes_manifest" "service_monitor" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "llm-router-metrics"
      namespace = "default"
      labels = {
        app = "llm-inference-router"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          app = "llm-inference-router"
        }
      }
      endpoints = [{
        port     = "metrics"
        interval = "30s"
        path     = "/metrics"
      }]
    }
  }
}