"""Model performance drift detection for routing decisions."""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Optional

from .metrics import MetricsCollector
from .models import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """Metrics for drift detection."""
    accuracy_drift: float
    latency_drift: float
    error_rate_drift: float
    confidence_score: float


class DriftDetector:
    """Detects performance drift in model routing decisions."""
    
    def __init__(self, window_size: int = 100, drift_threshold: float = 0.15):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Rolling windows for metrics
        self.accuracy_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.error_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Baseline metrics (first window)
        self.baselines: Dict[str, Dict] = {}
        
        logger.info(f"Drift detector initialized: window_size={window_size}, threshold={drift_threshold}")
    
    def record_routing_outcome(self, model_id: str, latency: float, 
                             accuracy: float, error_occurred: bool) -> None:
        """Record outcome for drift analysis."""
        self.accuracy_history[model_id].append(accuracy)
        self.latency_history[model_id].append(latency)
        self.error_history[model_id].append(1.0 if error_occurred else 0.0)
        
        # Set baseline after first full window
        if len(self.accuracy_history[model_id]) == self.window_size and model_id not in self.baselines:
            self.baselines[model_id] = {
                'accuracy': sum(self.accuracy_history[model_id]) / self.window_size,
                'latency': sum(self.latency_history[model_id]) / self.window_size,
                'error_rate': sum(self.error_history[model_id]) / self.window_size
            }
            logger.info(f"Baseline established for {model_id}: {self.baselines[model_id]}")
    
    def detect_drift(self, model_id: str) -> Optional[DriftMetrics]:
        """Detect if model performance has drifted from baseline."""
        if (model_id not in self.baselines or 
            len(self.accuracy_history[model_id]) < self.window_size):
            return None
        
        # Current window averages
        current_accuracy = sum(self.accuracy_history[model_id]) / self.window_size
        current_latency = sum(self.latency_history[model_id]) / self.window_size
        current_error_rate = sum(self.error_history[model_id]) / self.window_size
        
        baseline = self.baselines[model_id]
        
        # Calculate drift percentages
        accuracy_drift = abs(current_accuracy - baseline['accuracy']) / max(baseline['accuracy'], 0.01)
        latency_drift = abs(current_latency - baseline['latency']) / max(baseline['latency'], 0.01)
        error_rate_drift = abs(current_error_rate - baseline['error_rate']) / max(baseline['error_rate'], 0.01)
        
        # Confidence based on sample size and variance
        confidence_score = min(1.0, len(self.accuracy_history[model_id]) / self.window_size)
        
        metrics = DriftMetrics(
            accuracy_drift=accuracy_drift,
            latency_drift=latency_drift, 
            error_rate_drift=error_rate_drift,
            confidence_score=confidence_score
        )
        
        # Log if significant drift detected
        max_drift = max(accuracy_drift, latency_drift, error_rate_drift)
        if max_drift > self.drift_threshold:
            logger.warning(f"Performance drift detected for {model_id}: {max_drift:.2%}")
        
        return metrics
    
    def get_drift_status(self) -> Dict[str, bool]:
        """Get drift status for all monitored models."""
        status = {}
        for model_id in self.baselines.keys():
            metrics = self.detect_drift(model_id)
            if metrics:
                max_drift = max(metrics.accuracy_drift, metrics.latency_drift, metrics.error_rate_drift)
                status[model_id] = max_drift > self.drift_threshold
            else:
                status[model_id] = False
        return status
