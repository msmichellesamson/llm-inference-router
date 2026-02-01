from typing import Dict, List, Optional
import time
import asyncio
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class EndpointHealth:
    url: str
    response_time: float
    error_rate: float
    last_check: float
    active_requests: int = 0

class AdaptiveLoadBalancer:
    """Intelligent load balancer that routes requests based on endpoint health."""
    
    def __init__(self, health_check_interval: int = 30):
        self.endpoints: Dict[str, List[EndpointHealth]] = defaultdict(list)
        self.health_check_interval = health_check_interval
        self._running = False
    
    def register_endpoint(self, model_type: str, url: str) -> None:
        """Register a new endpoint for a model type."""
        endpoint = EndpointHealth(
            url=url,
            response_time=0.0,
            error_rate=0.0,
            last_check=time.time()
        )
        self.endpoints[model_type].append(endpoint)
    
    def get_best_endpoint(self, model_type: str) -> Optional[str]:
        """Select the best endpoint based on response time and error rate."""
        if model_type not in self.endpoints:
            return None
        
        healthy_endpoints = [
            ep for ep in self.endpoints[model_type]
            if ep.error_rate < 0.1 and ep.response_time < 5.0
        ]
        
        if not healthy_endpoints:
            return None
        
        # Weighted scoring: lower is better
        best_endpoint = min(
            healthy_endpoints,
            key=lambda ep: (ep.response_time * 0.7) + 
                          (ep.error_rate * 0.2) + 
                          (ep.active_requests * 0.1)
        )
        
        best_endpoint.active_requests += 1
        return best_endpoint.url
    
    def update_endpoint_metrics(self, url: str, response_time: float, 
                               success: bool) -> None:
        """Update endpoint performance metrics."""
        for endpoints in self.endpoints.values():
            for endpoint in endpoints:
                if endpoint.url == url:
                    endpoint.response_time = response_time
                    endpoint.error_rate = 0.0 if success else 1.0
                    endpoint.last_check = time.time()
                    endpoint.active_requests = max(0, endpoint.active_requests - 1)
                    break
    
    async def start_health_checks(self) -> None:
        """Start background health checking."""
        self._running = True
        while self._running:
            await asyncio.sleep(self.health_check_interval)
            # Health check logic would go here
    
    def stop(self) -> None:
        """Stop the load balancer."""
        self._running = False