import pytest
from unittest.mock import Mock, patch
from src.core.load_balancer import LoadBalancer, LoadBalancerStrategy
from src.core.models import ModelEndpoint
from src.core.exceptions import NoHealthyEndpointsError


class TestLoadBalancer:
    @pytest.fixture
    def endpoints(self):
        return [
            ModelEndpoint(name="model1", url="http://model1:8000", priority=1),
            ModelEndpoint(name="model2", url="http://model2:8000", priority=2),
            ModelEndpoint(name="model3", url="http://model3:8000", priority=1)
        ]
    
    @pytest.fixture
    def load_balancer(self, endpoints):
        return LoadBalancer(endpoints)
    
    def test_round_robin_selection(self, load_balancer):
        """Test round-robin endpoint selection"""
        load_balancer.strategy = LoadBalancerStrategy.ROUND_ROBIN
        
        # All endpoints healthy
        load_balancer.healthy_endpoints = load_balancer.endpoints.copy()
        
        selected = []
        for _ in range(6):
            endpoint = load_balancer.select_endpoint()
            selected.append(endpoint.name)
        
        # Should cycle through endpoints
        assert selected == ["model1", "model2", "model3", "model1", "model2", "model3"]
    
    def test_priority_selection(self, load_balancer):
        """Test priority-based endpoint selection"""
        load_balancer.strategy = LoadBalancerStrategy.PRIORITY
        load_balancer.healthy_endpoints = load_balancer.endpoints.copy()
        
        # Should always select highest priority (lowest number)
        for _ in range(3):
            endpoint = load_balancer.select_endpoint()
            assert endpoint.priority == 1
    
    def test_no_healthy_endpoints(self, load_balancer):
        """Test behavior when no endpoints are healthy"""
        load_balancer.healthy_endpoints = []
        
        with pytest.raises(NoHealthyEndpointsError):
            load_balancer.select_endpoint()
    
    def test_mark_unhealthy(self, load_balancer):
        """Test marking endpoints as unhealthy"""
        load_balancer.healthy_endpoints = load_balancer.endpoints.copy()
        initial_count = len(load_balancer.healthy_endpoints)
        
        load_balancer.mark_unhealthy("model1")
        assert len(load_balancer.healthy_endpoints) == initial_count - 1
        assert not any(e.name == "model1" for e in load_balancer.healthy_endpoints)
    
    def test_mark_healthy(self, load_balancer):
        """Test marking endpoints as healthy"""
        load_balancer.healthy_endpoints = []
        
        load_balancer.mark_healthy("model1")
        assert len(load_balancer.healthy_endpoints) == 1
        assert load_balancer.healthy_endpoints[0].name == "model1"
    
    def test_get_endpoint_stats(self, load_balancer):
        """Test endpoint statistics"""
        stats = load_balancer.get_endpoint_stats()
        
        assert "total_endpoints" in stats
        assert "healthy_endpoints" in stats
        assert "unhealthy_endpoints" in stats
        assert stats["total_endpoints"] == 3
        assert stats["healthy_endpoints"] == 3
        assert stats["unhealthy_endpoints"] == 0
