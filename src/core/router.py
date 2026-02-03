import logging
from typing import Dict, List, Optional
from .models import LLMEndpoint, RoutingRequest, RoutingResponse
from .complexity_analyzer import ComplexityAnalyzer
from .load_balancer import LoadBalancer
from .circuit_breaker import CircuitBreakerManager
from .metrics import MetricsCollector

logger = logging.getLogger(__name__)

class LLMRouter:
    def __init__(self):
        self.endpoints: Dict[str, LLMEndpoint] = {}
        self.complexity_analyzer = ComplexityAnalyzer()
        self.load_balancer = LoadBalancer()
        self.circuit_breaker = CircuitBreakerManager()
        self.metrics = MetricsCollector()
    
    def register_endpoint(self, endpoint: LLMEndpoint):
        """Register a new LLM endpoint"""
        self.endpoints[endpoint.name] = endpoint
        logger.info(f"Registered endpoint: {endpoint.name}")
    
    async def route_request(self, request: RoutingRequest) -> RoutingResponse:
        """Route request to optimal endpoint with circuit breaker protection"""
        try:
            # Analyze complexity
            complexity = await self.complexity_analyzer.analyze(request.prompt)
            
            # Filter available endpoints by circuit breaker state
            available_endpoints = [
                ep for ep in self.endpoints.values() 
                if self.circuit_breaker.is_available(ep.name)
            ]
            
            if not available_endpoints:
                raise Exception("No available endpoints - all circuit breakers open")
            
            # Select optimal endpoint
            selected = self.load_balancer.select_endpoint(
                available_endpoints, complexity, request.priority
            )
            
            # Execute request with circuit breaker
            start_time = self.metrics.start_timer()
            
            try:
                response = await self._execute_request(selected, request)
                self.circuit_breaker.record_success(selected.name)
                
                # Record metrics
                self.metrics.record_request(
                    endpoint=selected.name,
                    latency=self.metrics.end_timer(start_time),
                    tokens=len(response.content.split()),
                    cost=selected.cost_per_token * len(response.content.split())
                )
                
                return response
                
            except Exception as e:
                self.circuit_breaker.record_failure(selected.name)
                logger.error(f"Request failed for {selected.name}: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            raise
    
    async def _execute_request(self, endpoint: LLMEndpoint, request: RoutingRequest) -> RoutingResponse:
        """Execute request against specific endpoint"""
        # This would integrate with actual LLM APIs
        # For now, return mock response
        return RoutingResponse(
            content=f"Mock response from {endpoint.name}",
            endpoint_used=endpoint.name,
            latency_ms=endpoint.avg_latency,
            cost=endpoint.cost_per_token * 100
        )
    
    def get_health(self) -> Dict:
        """Get router health including circuit breaker status"""
        return {
            "endpoints": len(self.endpoints),
            "circuit_breakers": self.circuit_breaker.get_status(),
            "metrics": self.metrics.get_summary()
        }