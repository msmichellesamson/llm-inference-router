import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

from src.core.drift_detector import DriftDetector, DriftMetrics
from src.core.exceptions import DriftDetectionError


@pytest.fixture
def drift_detector():
    return DriftDetector(
        window_size=10,
        threshold=0.1,
        min_samples=5
    )


@pytest.mark.asyncio
async def test_drift_detector_initialization(drift_detector):
    """Test drift detector initializes with correct parameters."""
    assert drift_detector.window_size == 10
    assert drift_detector.threshold == 0.1
    assert drift_detector.min_samples == 5
    assert len(drift_detector.baseline_metrics) == 0


@pytest.mark.asyncio
async def test_add_sample(drift_detector):
    """Test adding samples to drift detector."""
    metrics = DriftMetrics(
        response_time=0.5,
        complexity_score=0.7,
        model_confidence=0.9,
        timestamp=datetime.now()
    )
    
    await drift_detector.add_sample(metrics)
    assert len(drift_detector.baseline_metrics) == 1


@pytest.mark.asyncio
async def test_detect_no_drift_insufficient_samples(drift_detector):
    """Test drift detection with insufficient samples."""
    for i in range(3):
        metrics = DriftMetrics(
            response_time=0.5,
            complexity_score=0.7,
            model_confidence=0.9,
            timestamp=datetime.now()
        )
        await drift_detector.add_sample(metrics)
    
    is_drift = await drift_detector.detect_drift()
    assert not is_drift


@pytest.mark.asyncio
async def test_detect_drift_significant_change(drift_detector):
    """Test drift detection with significant performance change."""
    # Add baseline samples
    for i in range(6):
        metrics = DriftMetrics(
            response_time=0.5,
            complexity_score=0.7,
            model_confidence=0.9,
            timestamp=datetime.now()
        )
        await drift_detector.add_sample(metrics)
    
    # Add drifted samples
    for i in range(6):
        metrics = DriftMetrics(
            response_time=2.0,  # Significant increase
            complexity_score=0.3,  # Significant decrease
            model_confidence=0.5,  # Significant decrease
            timestamp=datetime.now()
        )
        await drift_detector.add_sample(metrics)
    
    is_drift = await drift_detector.detect_drift()
    assert is_drift


@pytest.mark.asyncio
async def test_window_size_management(drift_detector):
    """Test that detector maintains window size limit."""
    # Add more samples than window size
    for i in range(15):
        metrics = DriftMetrics(
            response_time=0.5 + i * 0.1,
            complexity_score=0.7,
            model_confidence=0.9,
            timestamp=datetime.now()
        )
        await drift_detector.add_sample(metrics)
    
    # Should maintain only window_size samples
    assert len(drift_detector.baseline_metrics) == drift_detector.window_size


@pytest.mark.asyncio
async def test_get_drift_metrics(drift_detector):
    """Test retrieving drift statistics."""
    for i in range(8):
        metrics = DriftMetrics(
            response_time=0.5 + i * 0.1,
            complexity_score=0.7 - i * 0.05,
            model_confidence=0.9 - i * 0.02,
            timestamp=datetime.now()
        )
        await drift_detector.add_sample(metrics)
    
    stats = await drift_detector.get_drift_metrics()
    
    assert 'mean_response_time' in stats
    assert 'std_response_time' in stats
    assert 'drift_score' in stats
    assert stats['sample_count'] == 8