"""Enhanced router with structured logging."""

from typing import Dict, List, Optional, Tuple, Any
import asyncio
from dataclasses import dataclass

from .models import ModelConfig, RoutingDecision, QueryRequest
from .complexity_analyzer import ComplexityAnalyzer
from .circuit_breaker import CircuitBreaker
from .load_balancer import LoadBalancer
from .health_checker import HealthChecker
from .exceptions import RoutingError, ModelUnavailableError
from .logger import get_logger, log_with_context, set_correlation_id

@dataclass
class RouterMetrics:
    """Router performance metrics."""
    total_requests: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    avg_latency: float = 0.0

class IntelligentRouter:
    """Multi-model LLM router with intelligent routing decisions."""
    
    def __init__(self, models: List[ModelConfig]):
        self.models = {model.name: model for model in models}
        self.complexity_analyzer = ComplexityAnalyzer()
        self.circuit_breakers = {name: CircuitBreaker() for name in self.models}
        self.load_balancer = LoadBalancer()
        self.health_checker = HealthChecker(list(self.models.keys()))
        self.metrics = RouterMetrics()
        self.logger = get_logger(__name__)
    
    async def route_request(self, request: QueryRequest) -> RoutingDecision:
        """Route request to optimal model based on complexity and availability."""
        request_id = set_correlation_id(request.request_id)
        
        log_with_context(
            self.logger, 20, "Processing routing request",
            query_length=len(request.query),
            model_count=len(self.models)
        )
        
        try:
            # Analyze query complexity
            complexity = await self.complexity_analyzer.analyze(request.query)
            
            # Get available models
            available_models = await self._get_available_models()
            
            if not available_models:
                raise ModelUnavailableError("No models available")
            
            # Select optimal model
            selected_model = self._select_model(complexity, available_models, request)
            
            decision = RoutingDecision(
                model_name=selected_model.name,
                confidence=complexity.confidence,
                estimated_cost=selected_model.cost_per_token * complexity.token_count,
                estimated_latency=selected_model.avg_latency,
                reasoning=f"Selected based on complexity: {complexity.level}"
            )
            
            self.metrics.successful_routes += 1
            
            log_with_context(
                self.logger, 20, "Request routed successfully",
                selected_model=selected_model.name,
                complexity_level=complexity.level,
                estimated_cost=decision.estimated_cost
            )
            
            return decision
            
        except Exception as e:
            self.metrics.failed_routes += 1
            log_with_context(
                self.logger, 40, "Routing failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise RoutingError(f"Failed to route request: {str(e)}") from e
        
        finally:
            self.metrics.total_requests += 1
    
    async def _get_available_models(self) -> List[ModelConfig]:
        """Get list of healthy, available models."""
        available = []
        
        for name, model in self.models.items():
            if (self.circuit_breakers[name].is_available() and
                await self.health_checker.is_healthy(name)):
                available.append(model)
        
        return available
    
    def _select_model(self, complexity, available_models: List[ModelConfig], 
                     request: QueryRequest) -> ModelConfig:
        """Select optimal model based on complexity and constraints."""
        # Filter by complexity requirements
        suitable_models = [
            model for model in available_models
            if model.min_complexity <= complexity.level <= model.max_complexity
        ]
        
        if not suitable_models:
            suitable_models = available_models  # Fallback to any available
        
        # Apply load balancing
        return self.load_balancer.select(suitable_models)
