import pytest
import asyncio
import json
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

import structlog
from redis.asyncio import Redis
from prometheus_client import REGISTRY

from src.core.router import LLMRouter, RoutingDecision, ModelType
from src.core.models import ComplexityAnalyzer, QueryComplexity, ModelConfig
from src.core.metrics import MetricsCollector
from src.database.redis_cache import RedisCache


logger = structlog.get_logger(__name__)


class TestLLMRouterIntegration:
    """Integration tests for LLM routing logic with real dependencies."""
    
    @pytest.fixture
    async def redis_client(self) -> Redis:
        """Redis client for testing."""
        client = Redis(host="localhost", port=6379, db=15)
        await client.flushdb()
        yield client
        await client.flushdb()
        await client.aclose()
    
    @pytest.fixture
    async def redis_cache(self, redis_client: Redis) -> RedisCache:
        """Redis cache instance."""
        return RedisCache(redis_client)
    
    @pytest.fixture
    def metrics_collector(self) -> MetricsCollector:
        """Metrics collector with cleared registry."""
        # Clear existing metrics
        for collector in list(REGISTRY._collector_to_names.keys()):
            if hasattr(collector, '_name') and collector._name.startswith('llm_'):
                REGISTRY.unregister(collector)
        return MetricsCollector()
    
    @pytest.fixture
    def complexity_analyzer(self) -> ComplexityAnalyzer:
        """Complexity analyzer instance."""
        return ComplexityAnalyzer()
    
    @pytest.fixture
    def model_configs(self) -> Dict[str, ModelConfig]:
        """Test model configurations."""
        return {
            "local_small": ModelConfig(
                name="local_small",
                model_type=ModelType.LOCAL,
                cost_per_1k_tokens=0.0,
                avg_latency_ms=150,
                max_tokens=2048,
                complexity_threshold=0.3,
                enabled=True
            ),
            "local_medium": ModelConfig(
                name="local_medium", 
                model_type=ModelType.LOCAL,
                cost_per_1k_tokens=0.0,
                avg_latency_ms=400,
                max_tokens=4096,
                complexity_threshold=0.6,
                enabled=True
            ),
            "gpt-3.5-turbo": ModelConfig(
                name="gpt-3.5-turbo",
                model_type=ModelType.CLOUD,
                cost_per_1k_tokens=0.002,
                avg_latency_ms=800,
                max_tokens=4096,
                complexity_threshold=0.7,
                enabled=True
            ),
            "gpt-4": ModelConfig(
                name="gpt-4",
                model_type=ModelType.CLOUD,
                cost_per_1k_tokens=0.03,
                avg_latency_ms=1200,
                max_tokens=8192,
                complexity_threshold=0.9,
                enabled=True
            )
        }
    
    @pytest.fixture
    async def router(
        self, 
        redis_cache: RedisCache, 
        metrics_collector: MetricsCollector,
        complexity_analyzer: ComplexityAnalyzer,
        model_configs: Dict[str, ModelConfig]
    ) -> LLMRouter:
        """LLM router instance with all dependencies."""
        router = LLMRouter(
            redis_cache=redis_cache,
            metrics_collector=metrics_collector,
            complexity_analyzer=complexity_analyzer,
            model_configs=model_configs
        )
        await router.initialize()
        return router

    async def test_simple_query_routes_to_local_small(
        self, router: LLMRouter, metrics_collector: MetricsCollector
    ):
        """Simple queries should route to smallest local model."""
        query = "What is 2+2?"
        
        decision = await router.route_query(query, user_id="test_user")
        
        assert decision.model_name == "local_small"
        assert decision.model_type == ModelType.LOCAL
        assert decision.reasoning == "Low complexity query routed to fastest local model"
        assert decision.estimated_cost == 0.0
        assert decision.estimated_latency_ms == 150
        
        # Check metrics were recorded
        routing_counter = metrics_collector.routing_decisions_total
        assert routing_counter._value._value > 0

    async def test_medium_complexity_routes_appropriately(
        self, router: LLMRouter
    ):
        """Medium complexity queries should route to appropriate model."""
        query = """Explain the differences between microservices and monolithic 
        architecture, including pros and cons of each approach."""
        
        decision = await router.route_query(query, user_id="test_user")
        
        assert decision.model_name in ["local_medium", "gpt-3.5-turbo"]
        assert decision.complexity_score > 0.3
        
    async def test_high_complexity_routes_to_best_model(
        self, router: LLMRouter
    ):
        """High complexity queries should route to most capable model."""
        query = """Write a detailed analysis of quantum computing algorithms, 
        including Shor's algorithm, Grover's algorithm, and their implications 
        for cryptography. Include mathematical formulations and complexity analysis."""
        
        decision = await router.route_query(query, user_id="test_user")
        
        assert decision.model_name in ["gpt-4", "local_medium"]
        assert decision.complexity_score > 0.6

    async def test_user_preference_overrides_default_routing(
        self, router: LLMRouter
    ):
        """User preferences should override default routing decisions."""
        query = "Simple question"
        preferences = {"prefer_local": False, "max_cost": 0.05}
        
        decision = await router.route_query(
            query, 
            user_id="test_user", 
            user_preferences=preferences
        )
        
        # Should route to cloud model despite low complexity
        assert decision.model_type == ModelType.CLOUD

    async def test_cost_constraint_respected(self, router: LLMRouter):
        """Cost constraints should be respected in routing decisions."""
        query = "Complex analysis requiring detailed reasoning"
        preferences = {"max_cost": 0.001}  # Very low cost limit
        
        decision = await router.route_query(
            query,
            user_id="test_user",
            user_preferences=preferences
        )
        
        assert decision.estimated_cost <= 0.001
        assert decision.model_type == ModelType.LOCAL

    async def test_latency_constraint_respected(self, router: LLMRouter):
        """Latency constraints should be respected in routing decisions."""
        query = "Any question"
        preferences = {"max_latency_ms": 200}
        
        decision = await router.route_query(
            query,
            user_id="test_user", 
            user_preferences=preferences
        )
        
        assert decision.estimated_latency_ms <= 200
        assert decision.model_name == "local_small"

    async def test_fallback_when_preferred_model_unavailable(
        self, router: LLMRouter, model_configs: Dict[str, ModelConfig]
    ):
        """Should fallback when preferred model is unavailable."""
        # Disable the small local model
        model_configs["local_small"].enabled = False
        
        query = "Simple question"
        
        decision = await router.route_query(query, user_id="test_user")
        
        assert decision.model_name != "local_small"
        assert decision.model_name in ["local_medium", "gpt-3.5-turbo"]

    async def test_caching_reduces_repeated_analysis(
        self, router: LLMRouter, redis_cache: RedisCache
    ):
        """Repeated queries should use cached complexity analysis."""
        query = "What is machine learning?"
        user_id = "test_user"
        
        # First request - should analyze and cache
        decision1 = await router.route_query(query, user_id=user_id)
        
        # Check cache was populated
        cache_key = f"complexity:{hash(query)}"
        cached_complexity = await redis_cache.get(cache_key)
        assert cached_complexity is not None
        
        # Second request - should use cache
        decision2 = await router.route_query(query, user_id=user_id)
        
        assert decision1.model_name == decision2.model_name
        assert decision1.complexity_score == decision2.complexity_score

    async def test_concurrent_requests_handled_correctly(
        self, router: LLMRouter
    ):
        """Multiple concurrent requests should be handled correctly."""
        queries = [
            "Simple question 1",
            "Simple question 2", 
            "Complex analysis of distributed systems architecture",
            "What is Python?",
            "Explain quantum entanglement in detail"
        ]
        
        # Execute all queries concurrently
        tasks = [
            router.route_query(query, user_id=f"user_{i}")
            for i, query in enumerate(queries)
        ]
        
        decisions = await asyncio.gather(*tasks)
        
        assert len(decisions) == 5
        for decision in decisions:
            assert isinstance(decision, RoutingDecision)
            assert decision.model_name is not None
            assert decision.complexity_score >= 0

    async def test_model_health_affects_routing(
        self, router: LLMRouter
    ):
        """Model health should affect routing decisions."""
        query = "Medium complexity question about databases"
        
        # Mark local_medium as unhealthy
        await router.update_model_health("local_medium", healthy=False)
        
        decision = await router.route_query(query, user_id="test_user")
        
        # Should not route to unhealthy model
        assert decision.model_name != "local_medium"

    async def test_metrics_recorded_for_all_decisions(
        self, router: LLMRouter, metrics_collector: MetricsCollector
    ):
        """All routing decisions should be recorded in metrics."""
        queries = ["Simple", "Medium complexity", "Very complex analysis"]
        
        initial_count = metrics_collector.routing_decisions_total._value._value
        
        for query in queries:
            await router.route_query(query, user_id="test_user")
        
        final_count = metrics_collector.routing_decisions_total._value._value
        assert final_count == initial_count + len(queries)

    async def test_token_estimation_accuracy(
        self, router: LLMRouter, complexity_analyzer: ComplexityAnalyzer
    ):
        """Token estimation should be reasonably accurate."""
        short_query = "Hi"
        long_query = "This is a much longer query " * 20
        
        short_decision = await router.route_query(short_query, user_id="test_user")
        long_decision = await router.route_query(long_query, user_id="test_user")
        
        # Longer query should have higher token estimate
        assert short_decision.estimated_tokens < long_decision.estimated_tokens

    async def test_routing_decision_serialization(
        self, router: LLMRouter
    ):
        """Routing decisions should be properly serializable."""
        query = "Test serialization"
        
        decision = await router.route_query(query, user_id="test_user")
        
        # Should be able to convert to dict and back
        decision_dict = decision.to_dict()
        assert isinstance(decision_dict, dict)
        assert decision_dict["model_name"] == decision.model_name
        assert decision_dict["complexity_score"] == decision.complexity_score

    @pytest.mark.parametrize("complexity_score,expected_models", [
        (0.1, ["local_small"]),
        (0.4, ["local_small", "local_medium"]),
        (0.7, ["local_medium", "gpt-3.5-turbo", "gpt-4"]),
        (0.95, ["gpt-4"])
    ])
    async def test_complexity_based_model_selection(
        self,
        router: LLMRouter,
        complexity_score: float,
        expected_models: List[str]
    ):
        """Model selection should match complexity thresholds."""
        # Mock complexity analyzer to return specific score
        with patch.object(router.complexity_analyzer, 'analyze_query') as mock_analyze:
            mock_analyze.return_value = QueryComplexity(
                score=complexity_score,
                factors={"length": 0.5, "technical_terms": 0.3},
                estimated_tokens=100,
                reasoning="Test complexity"
            )
            
            decision = await router.route_query("test", user_id="test_user")
            
            assert decision.model_name in expected_models

    async def test_error_handling_with_invalid_preferences(
        self, router: LLMRouter
    ):
        """Invalid user preferences should not crash routing."""
        query = "Test query"
        invalid_preferences = {
            "max_cost": -1,  # Invalid negative cost
            "max_latency_ms": "not_a_number",  # Invalid type
            "unknown_preference": "value"  # Unknown preference
        }
        
        # Should not raise exception
        decision = await router.route_query(
            query,
            user_id="test_user",
            user_preferences=invalid_preferences
        )
        
        assert isinstance(decision, RoutingDecision)

    async def test_cache_expiration_handling(
        self, router: LLMRouter, redis_cache: RedisCache
    ):
        """Expired cache entries should be handled gracefully."""
        query = "Cache expiration test"
        
        # Set very short TTL
        original_ttl = redis_cache.default_ttl
        redis_cache.default_ttl = 1
        
        try:
            decision1 = await router.route_query(query, user_id="test_user")
            
            # Wait for cache to expire
            await asyncio.sleep(2)
            
            decision2 = await router.route_query(query, user_id="test_user")
            
            # Both decisions should be valid
            assert isinstance(decision1, RoutingDecision)
            assert isinstance(decision2, RoutingDecision)
            
        finally:
            redis_cache.default_ttl = original_ttl