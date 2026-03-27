import asyncio
import time
from typing import Dict, Any, Optional
from .models import ModelType, RoutingDecision, QueryRequest
from .complexity_analyzer import ComplexityAnalyzer
from .load_balancer import LoadBalancer
from .circuit_breaker import CircuitBreaker
from .exceptions import RouterError, ModelTimeoutError
from .metrics import RouterMetrics
import logging

logger = logging.getLogger(__name__)

class LLMRouter:
    def __init__(self, config: Dict[str, Any]):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.load_balancer = LoadBalancer(config.get('models', {}))
        self.circuit_breaker = CircuitBreaker()
        self.metrics = RouterMetrics()
        self.default_timeout = config.get('timeout_seconds', 30)
        self.fallback_timeout = config.get('fallback_timeout_seconds', 10)
        
    async def route_query(self, request: QueryRequest) -> RoutingDecision:
        """Route query with timeout handling and graceful fallback"""
        start_time = time.time()
        
        try:
            # Analyze complexity with timeout
            complexity_task = asyncio.create_task(
                self._analyze_with_timeout(request)
            )
            
            complexity_score = await asyncio.wait_for(
                complexity_task, 
                timeout=self.default_timeout
            )
            
            # Get available models
            available_models = self.load_balancer.get_available_models()
            
            # Make routing decision
            decision = self._make_routing_decision(
                complexity_score, 
                available_models, 
                request
            )
            
            # Record metrics
            self.metrics.record_routing_decision(
                decision.model_type,
                complexity_score,
                time.time() - start_time
            )
            
            logger.info(f"Routed query to {decision.model_type.value} "
                       f"(complexity: {complexity_score:.2f})")
            
            return decision
            
        except asyncio.TimeoutError:
            logger.warning(f"Router timeout after {self.default_timeout}s, using fallback")
            return await self._fallback_routing(request)
            
        except Exception as e:
            logger.error(f"Router error: {e}")
            self.metrics.record_error('routing_error')
            raise RouterError(f"Failed to route query: {e}")
    
    async def _analyze_with_timeout(self, request: QueryRequest) -> float:
        """Analyze complexity with built-in timeout protection"""
        try:
            return await self.complexity_analyzer.analyze(request.query)
        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}")
            # Return medium complexity as safe default
            return 0.5
    
    async def _fallback_routing(self, request: QueryRequest) -> RoutingDecision:
        """Fast fallback routing when primary analysis times out"""
        try:
            # Quick heuristic: route short queries to local, long to cloud
            is_simple = len(request.query.split()) < 20
            
            available_models = self.load_balancer.get_available_models()
            
            if is_simple and ModelType.LOCAL in available_models:
                model_type = ModelType.LOCAL
            else:
                model_type = ModelType.CLOUD
            
            decision = RoutingDecision(
                model_type=model_type,
                model_name=available_models[model_type][0],
                confidence=0.6,  # Lower confidence for fallback
                estimated_cost=0.001 if model_type == ModelType.LOCAL else 0.01,
                estimated_latency=1.0 if model_type == ModelType.LOCAL else 3.0
            )
            
            self.metrics.record_fallback_routing()
            logger.info(f"Used fallback routing to {model_type.value}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Fallback routing failed: {e}")
            raise RouterError(f"All routing methods failed: {e}")
    
    def _make_routing_decision(
        self, 
        complexity: float, 
        available_models: Dict[ModelType, list],
        request: QueryRequest
    ) -> RoutingDecision:
        """Make routing decision based on complexity and availability"""
        # Route complex queries to cloud models
        if complexity > 0.7 and ModelType.CLOUD in available_models:
            model_type = ModelType.CLOUD
            estimated_cost = complexity * 0.02
            estimated_latency = 2.0 + complexity * 2.0
        # Route simple queries to local models when available
        elif complexity < 0.3 and ModelType.LOCAL in available_models:
            model_type = ModelType.LOCAL
            estimated_cost = 0.001
            estimated_latency = 0.5 + complexity * 1.0
        # Default to cloud for medium complexity or when local unavailable
        else:
            model_type = ModelType.CLOUD if ModelType.CLOUD in available_models else ModelType.LOCAL
            estimated_cost = 0.01
            estimated_latency = 3.0
        
        model_name = available_models[model_type][0]
        
        return RoutingDecision(
            model_type=model_type,
            model_name=model_name,
            confidence=min(0.9, 0.5 + complexity),
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency
        )
