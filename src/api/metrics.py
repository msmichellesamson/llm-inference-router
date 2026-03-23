from fastapi import APIRouter
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import PlainTextResponse
import time
from typing import Dict

router = APIRouter()

# Prometheus metrics
ROUTING_DECISIONS = Counter(
    'llm_routing_decisions_total',
    'Total routing decisions made',
    ['model', 'complexity_level', 'routing_reason']
)

REQUEST_DURATION = Histogram(
    'llm_request_duration_seconds',
    'Time spent processing requests',
    ['model', 'status']
)

ACTIVE_REQUESTS = Gauge(
    'llm_active_requests',
    'Number of active requests per model',
    ['model']
)

CACHE_HITS = Counter(
    'llm_cache_hits_total',
    'Cache hit/miss counts',
    ['status']
)

class MetricsCollector:
    def __init__(self):
        self.start_times: Dict[str, float] = {}
    
    def record_routing_decision(self, model: str, complexity: str, reason: str):
        """Record a routing decision made by the system"""
        ROUTING_DECISIONS.labels(
            model=model, 
            complexity_level=complexity, 
            routing_reason=reason
        ).inc()
    
    def start_request_timer(self, request_id: str, model: str):
        """Start timing a request"""
        self.start_times[request_id] = time.time()
        ACTIVE_REQUESTS.labels(model=model).inc()
    
    def end_request_timer(self, request_id: str, model: str, status: str):
        """End timing a request and record duration"""
        if request_id in self.start_times:
            duration = time.time() - self.start_times[request_id]
            REQUEST_DURATION.labels(model=model, status=status).observe(duration)
            del self.start_times[request_id]
        ACTIVE_REQUESTS.labels(model=model).dec()
    
    def record_cache_hit(self, hit: bool):
        """Record cache hit or miss"""
        status = 'hit' if hit else 'miss'
        CACHE_HITS.labels(status=status).inc()

# Global metrics collector instance
metrics = MetricsCollector()

@router.get("/metrics")
def get_metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(generate_latest(), media_type="text/plain")