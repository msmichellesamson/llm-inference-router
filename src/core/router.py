import logging
from typing import List, Dict, Any, Optional
from .models import ModelProvider, RoutingDecision, QueryRequest
from .complexity_analyzer import ComplexityAnalyzer
from .load_balancer import LoadBalancer
from .circuit_breaker import CircuitBreaker
from .exceptions import RoutingError, ModelUnavailableError
from .context import get_correlation_id, get_context

logger = logging.getLogger(__name__)

class LLMRouter:
    """Routes queries to optimal LLM providers based on complexity and availability."""
    
    def __init__(self, providers: List[ModelProvider]):
        self.providers = {p.name: p for p in providers}
        self.complexity_analyzer = ComplexityAnalyzer()
        self.load_balancer = LoadBalancer(providers)
        self.circuit_breakers = {
            p.name: CircuitBreaker(failure_threshold=5, timeout=60)
            for p in providers
        }
    
    async def route_query(self, request: QueryRequest) -> RoutingDecision:
        """Route query to best available provider."""
        correlation_id = get_correlation_id()
        context = get_context()
        
        logger.info(
            f"Routing query [correlation_id={correlation_id}] "
            f"user_id={getattr(context, 'user_id', None)} "
            f"preference={getattr(context, 'model_preference', None)}"
        )
        
        try:
            # Analyze query complexity
            complexity = await self.complexity_analyzer.analyze(request.query)
            
            # Check user preference from context
            preferred_provider = None
            if context and context.model_preference:
                preferred_provider = context.model_preference
                logger.debug(f"User preference: {preferred_provider} [correlation_id={correlation_id}]")
            
            # Get available providers
            available_providers = self._get_healthy_providers()
            
            if not available_providers:
                raise ModelUnavailableError("No healthy providers available")
            
            # Select provider based on complexity and preference
            provider = self.load_balancer.select_provider(
                available_providers, 
                complexity,
                preferred=preferred_provider
            )
            
            logger.info(
                f"Selected provider: {provider.name} for complexity: {complexity.score} "
                f"[correlation_id={correlation_id}]"
            )
            
            return RoutingDecision(
                provider=provider,
                complexity=complexity,
                estimated_cost=self._estimate_cost(provider, request),
                estimated_latency=self._estimate_latency(provider, complexity)
            )
            
        except Exception as e:
            logger.error(f"Routing failed [correlation_id={correlation_id}]: {e}")
            raise RoutingError(f"Failed to route query: {str(e)}") from e
    
    def _get_healthy_providers(self) -> List[ModelProvider]:
        """Get providers with healthy circuit breakers."""
        return [
            provider for name, provider in self.providers.items()
            if not self.circuit_breakers[name].is_open
        ]
    
    def _estimate_cost(self, provider: ModelProvider, request: QueryRequest) -> float:
        """Estimate request cost."""
        token_count = len(request.query.split()) * 1.3  # Rough estimation
        return provider.cost_per_token * token_count
    
    def _estimate_latency(self, provider: ModelProvider, complexity) -> float:
        """Estimate request latency."""
        base_latency = provider.avg_latency_ms
        complexity_multiplier = 1.0 + (complexity.score * 0.5)
        return base_latency * complexity_multiplier
