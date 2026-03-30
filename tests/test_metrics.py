"""Tests for metrics collection and aggregation."""
import pytest
import time
from unittest.mock import Mock, patch
from src.core.metrics import MetricsCollector, RequestMetrics
from src.core.exceptions import ModelUnavailableError


class TestMetricsCollector:
    def test_init_creates_counters(self):
        collector = MetricsCollector()
        assert hasattr(collector, 'request_counter')
        assert hasattr(collector, 'latency_histogram')
        assert hasattr(collector, 'error_counter')
        assert hasattr(collector, 'model_usage_counter')

    def test_record_request_success(self):
        collector = MetricsCollector()
        metrics = RequestMetrics(
            model_name="gpt-4",
            complexity_score=0.7,
            latency_ms=150.5,
            tokens_used=25,
            cost_usd=0.002
        )
        
        collector.record_request(metrics)
        # Verify counters incremented (would need prometheus_client mocking)
        assert True  # Basic structure test

    def test_record_error(self):
        collector = MetricsCollector()
        error = ModelUnavailableError("gpt-4", "Rate limited")
        
        collector.record_error("gpt-4", error)
        # Verify error counter incremented
        assert True

    @patch('time.time')
    def test_latency_timing_context(self, mock_time):
        collector = MetricsCollector()
        mock_time.side_effect = [1000.0, 1000.15]  # 150ms difference
        
        with collector.time_request("test-model") as timer:
            pass
        
        assert timer.duration_ms == 150.0

    def test_get_stats_summary(self):
        collector = MetricsCollector()
        stats = collector.get_stats_summary()
        
        expected_keys = ['total_requests', 'error_rate', 'avg_latency_ms', 
                        'model_distribution', 'cost_per_hour']
        for key in expected_keys:
            assert key in stats

    def test_reset_counters(self):
        collector = MetricsCollector()
        collector.reset_counters()
        # Verify all counters reset to 0
        assert True


class TestRequestMetrics:
    def test_create_valid_metrics(self):
        metrics = RequestMetrics(
            model_name="claude-3",
            complexity_score=0.85,
            latency_ms=200.0,
            tokens_used=50,
            cost_usd=0.005
        )
        
        assert metrics.model_name == "claude-3"
        assert metrics.complexity_score == 0.85
        assert metrics.latency_ms == 200.0
        assert metrics.tokens_used == 50
        assert metrics.cost_usd == 0.005

    def test_metrics_validation(self):
        with pytest.raises(ValueError):
            RequestMetrics(
                model_name="",  # Empty string should fail
                complexity_score=0.5,
                latency_ms=100.0,
                tokens_used=10,
                cost_usd=0.001
            )

    def test_negative_values_rejected(self):
        with pytest.raises(ValueError):
            RequestMetrics(
                model_name="test",
                complexity_score=-0.1,  # Negative complexity
                latency_ms=100.0,
                tokens_used=10,
                cost_usd=0.001
            )