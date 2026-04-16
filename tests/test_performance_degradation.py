"""Tests for performance degradation detector."""
import pytest
from src.core.performance_degradation import PerformanceDegradationDetector
from src.core.models import ModelProvider


@pytest.fixture
def detector():
    return PerformanceDegradationDetector(window_size=10, degradation_threshold=0.5)


def test_initial_state(detector):
    """Test detector starts with no degradation."""
    degraded, reason = detector.check_degradation(ModelProvider.OPENAI_GPT4)
    assert not degraded
    assert reason is None


def test_record_normal_performance(detector):
    """Test recording normal performance doesn't trigger degradation."""
    provider = ModelProvider.OPENAI_GPT4
    
    # Record baseline performance
    for _ in range(15):
        detector.record_request(provider, latency=1.0, is_error=False)
    
    degraded, reason = detector.check_degradation(provider)
    assert not degraded


def test_latency_degradation(detector):
    """Test detection of latency degradation."""
    provider = ModelProvider.OPENAI_GPT4
    
    # Establish baseline with 1s latency
    for _ in range(15):
        detector.record_request(provider, latency=1.0, is_error=False)
    
    # Simulate latency spike (100% increase)
    for _ in range(10):
        detector.record_request(provider, latency=2.0, is_error=False)
    
    degraded, reason = detector.check_degradation(provider)
    assert degraded
    assert "Latency increased" in reason


def test_error_rate_degradation(detector):
    """Test detection of error rate degradation."""
    provider = ModelProvider.ANTHROPIC_CLAUDE
    
    # Establish baseline with no errors
    for _ in range(15):
        detector.record_request(provider, latency=1.0, is_error=False)
    
    # Simulate error spike
    for _ in range(10):
        detector.record_request(provider, latency=1.0, is_error=True)
    
    degraded, reason = detector.check_degradation(provider)
    assert degraded
    assert "Error rate" in reason


def test_performance_summary(detector):
    """Test getting performance summary."""
    provider = ModelProvider.LOCAL_OLLAMA
    
    # No data initially
    summary = detector.get_performance_summary(provider)
    assert summary is None
    
    # Add some data
    for i in range(15):
        detector.record_request(provider, latency=1.0 + i * 0.1, is_error=False)
    
    summary = detector.get_performance_summary(provider)
    assert summary is not None
    assert summary["provider"] == provider.value
    assert "baseline_latency" in summary
    assert "current_latency" in summary
