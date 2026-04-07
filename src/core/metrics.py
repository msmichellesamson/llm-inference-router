"""Enhanced metrics collection with error tracking."""

import time
from typing import Dict, Optional
from prometheus_client import Counter, Histogram, Gauge
from .error_tracker import ErrorTracker

# Prometheus metrics
request_count = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'status']
)

request_duration = Histogram(
    'llm_request_duration_seconds',
    'Request duration in seconds',
    ['model']
)

error_count = Counter(
    'llm_errors_total',
    'Total errors by category',
    ['model', 'error_category']
)

recovery_time = Histogram(
    'llm_error_recovery_seconds',
    'Time to recover from errors',
    ['model', 'error_category']
)

active_requests = Gauge(
    'llm_active_requests',
    'Currently active requests',
    ['model']
)

class MetricsCollector:
    """Enhanced metrics collector with error tracking."""
    
    def __init__(self):
        self.error_tracker = ErrorTracker()
        self._active_requests: Dict[str, int] = {}
    
    def record_request(self, model: str, duration: float, success: bool):
        """Record a request completion."""
        status = "success" if success else "error"
        request_count.labels(model=model, status=status).inc()
        request_duration.labels(model=model).observe(duration)
        
        # Update active requests
        if model in self._active_requests:
            self._active_requests[model] = max(0, self._active_requests[model] - 1)
            active_requests.labels(model=model).set(self._active_requests[model])
    
    def record_error(self, model: str, error: Exception, 
                    recovery_time: Optional[float] = None):
        """Record an error with categorization."""
        # Track in error tracker
        self.error_tracker.track_error(model, error, recovery_time)
        
        # Update Prometheus metrics
        category = self.error_tracker.categorize_error(error)
        error_count.labels(model=model, error_category=category.value).inc()
        
        if recovery_time is not None:
            recovery_time.labels(model=model, error_category=category.value).observe(
                recovery_time
            )
    
    def start_request(self, model: str):
        """Mark request as started."""
        if model not in self._active_requests:
            self._active_requests[model] = 0
        
        self._active_requests[model] += 1
        active_requests.labels(model=model).set(self._active_requests[model])
        
        return time.time()
    
    def get_error_summary(self) -> Dict:
        """Get comprehensive error summary."""
        return {
            "top_errors": self.error_tracker.get_top_errors(),
            "error_stats": {
                model: self.error_tracker.get_error_stats(model)
                for model in self._active_requests.keys()
            }
        }

# Global metrics instance
metrics = MetricsCollector()