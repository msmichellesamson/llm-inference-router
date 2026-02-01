from typing import Dict, List, Optional
import asyncio
import logging
from dataclasses import dataclass

from .models import ModelProvider, ModelConfig
from .complexity_analyzer import ComplexityAnalyzer, ComplexityScore
from .metrics import MetricsCollector
from .load_balancer import LoadBalancer

logger = logging.getLogger(__name__)

@dataclass
class RoutingDecision:
    provider: ModelProvider
    model: str
    complexity_score: ComplexityScore
    estimated_cost: float
    estimated_latency: float

class LLMRouter:
    """Intelligent LLM router that optimizes for cost and latency."""
    
    def __init__(self, models: List[ModelConfig]):
        self.models = {model.name: model for model in models}
        self.complexity_analyzer = ComplexityAnalyzer()
        self.metrics = MetricsCollector()
        self.load_balancer = LoadBalancer()
        
        # Route thresholds
        self.local_threshold = 0.3  # Use local for simple queries
        self.cloud_threshold = 0.7  # Use premium cloud for complex queries
    
    async def route_query(self, query: str, user_id: str) -> RoutingDecision:
        """Route query to optimal model based on complexity analysis."""
        try:
            # Analyze query complexity
            complexity = self.complexity_analyzer.analyze(query)
            
            # Select model based on complexity
            model = self._select_model(complexity)
            
            # Get cost and latency estimates
            estimated_cost = self._estimate_cost(model, complexity.token_count)
            estimated_latency = self._estimate_latency(model)
            
            decision = RoutingDecision(
                provider=model.provider,
                model=model.name,
                complexity_score=complexity,
                estimated_cost=estimated_cost,
                estimated_latency=estimated_latency
            )
            
            # Record routing decision
            self.metrics.record_routing_decision(decision, user_id)
            
            logger.info(
                f"Routed query to {model.name} "
                f"(complexity: {complexity.score:.2f}, "
                f"cost: ${estimated_cost:.4f})"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error routing query: {e}")
            # Fallback to default model
            fallback = self._get_fallback_model()
            return RoutingDecision(fallback.provider, fallback.name, complexity, 0.0, 0.0)
    
    def _select_model(self, complexity: ComplexityScore) -> ModelConfig:
        """Select optimal model based on complexity score."""
        if complexity.score <= self.local_threshold:
            # Simple queries -> local model
            return self._get_local_model()
        elif complexity.score >= self.cloud_threshold:
            # Complex queries -> premium cloud model
            return self._get_premium_model()
        else:
            # Medium complexity -> balanced cloud model
            return self._get_balanced_model()
    
    def _get_local_model(self) -> ModelConfig:
        """Get best available local model."""
        local_models = [m for m in self.models.values() if m.provider == ModelProvider.LOCAL]
        return local_models[0] if local_models else self._get_fallback_model()
    
    def _get_premium_model(self) -> ModelConfig:
        """Get premium cloud model for complex queries."""
        premium_models = [m for m in self.models.values() 
                         if m.provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC]]
        return premium_models[0] if premium_models else self._get_fallback_model()
    
    def _get_balanced_model(self) -> ModelConfig:
        """Get balanced model for medium complexity."""
        return list(self.models.values())[0]  # Simple selection for now
    
    def _get_fallback_model(self) -> ModelConfig:
        """Get fallback model when routing fails."""
        return list(self.models.values())[0]
    
    def _estimate_cost(self, model: ModelConfig, tokens: int) -> float:
        """Estimate query cost based on model and token count."""
        return model.cost_per_token * tokens
    
    def _estimate_latency(self, model: ModelConfig) -> float:
        """Estimate query latency based on model type."""
        return model.avg_latency_ms
