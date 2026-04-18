from typing import Dict, List, Optional, Any
import time
from datetime import datetime
from dataclasses import dataclass
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest
from src.core.logger import logger


@dataclass
class MetricDefinition:
    name: str
    help_text: str
    labels: List[str]
    metric_type: str  # 'gauge', 'counter', 'histogram'


class MetricsExporter:
    """Custom Prometheus metrics exporter for LLM routing telemetry."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self._init_default_metrics()
    
    def _init_default_metrics(self):
        """Initialize core routing metrics."""
        metrics_config = [
            MetricDefinition(
                "llm_route_decisions_total",
                "Total routing decisions made",
                ["model_type", "complexity", "decision"],
                "counter"
            ),
            MetricDefinition(
                "llm_model_response_time_seconds",
                "Model response time distribution",
                ["model", "provider"],
                "histogram"
            ),
            MetricDefinition(
                "llm_active_requests",
                "Currently active requests per model",
                ["model", "provider"],
                "gauge"
            ),
            MetricDefinition(
                "llm_cost_per_request_usd",
                "Estimated cost per request",
                ["model", "provider"],
                "histogram"
            )
        ]
        
        for metric_def in metrics_config:
            self._create_metric(metric_def)
    
    def _create_metric(self, metric_def: MetricDefinition):
        """Create and register a Prometheus metric."""
        try:
            if metric_def.metric_type == "counter":
                metric = Counter(
                    metric_def.name,
                    metric_def.help_text,
                    metric_def.labels,
                    registry=self.registry
                )
            elif metric_def.metric_type == "gauge":
                metric = Gauge(
                    metric_def.name,
                    metric_def.help_text,
                    metric_def.labels,
                    registry=self.registry
                )
            elif metric_def.metric_type == "histogram":
                metric = Histogram(
                    metric_def.name,
                    metric_def.help_text,
                    metric_def.labels,
                    registry=self.registry
                )
            else:
                raise ValueError(f"Unknown metric type: {metric_def.metric_type}")
            
            self.metrics[metric_def.name] = metric
            logger.info(f"Registered metric: {metric_def.name}")
            
        except Exception as e:
            logger.error(f"Failed to create metric {metric_def.name}: {e}")
    
    def increment_counter(self, metric_name: str, labels: Dict[str, str], value: float = 1.0):
        """Increment a counter metric."""
        if metric_name in self.metrics:
            self.metrics[metric_name].labels(**labels).inc(value)
    
    def set_gauge(self, metric_name: str, labels: Dict[str, str], value: float):
        """Set a gauge metric value."""
        if metric_name in self.metrics:
            self.metrics[metric_name].labels(**labels).set(value)
    
    def observe_histogram(self, metric_name: str, labels: Dict[str, str], value: float):
        """Observe a histogram metric."""
        if metric_name in self.metrics:
            self.metrics[metric_name].labels(**labels).observe(value)
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')
