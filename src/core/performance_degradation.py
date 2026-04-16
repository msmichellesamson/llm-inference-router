"""Model performance degradation detection."""
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from .models import ModelProvider


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model."""
    avg_latency: float
    error_rate: float
    throughput: float
    timestamp: float


class PerformanceDegradationDetector:
    """Detects when model performance significantly degrades."""
    
    def __init__(self, window_size: int = 50, degradation_threshold: float = 0.3):
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self._latencies: Dict[ModelProvider, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._errors: Dict[ModelProvider, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._baselines: Dict[ModelProvider, PerformanceMetrics] = {}
        self._last_check: Dict[ModelProvider, float] = {}
    
    def record_request(self, provider: ModelProvider, latency: float, is_error: bool) -> None:
        """Record a request result for performance tracking."""
        self._latencies[provider].append(latency)
        self._errors[provider].append(1 if is_error else 0)
        
        # Update baseline if we have enough data and it's been a while
        if (len(self._latencies[provider]) >= self.window_size and 
            (provider not in self._last_check or 
             time.time() - self._last_check[provider] > 300)):  # 5 minutes
            self._update_baseline(provider)
    
    def _update_baseline(self, provider: ModelProvider) -> None:
        """Update baseline performance metrics."""
        latencies = list(self._latencies[provider])
        errors = list(self._errors[provider])
        
        if not latencies:
            return
            
        avg_latency = sum(latencies) / len(latencies)
        error_rate = sum(errors) / len(errors)
        throughput = len(latencies) / (max(latencies) - min(latencies) + 1)
        
        self._baselines[provider] = PerformanceMetrics(
            avg_latency=avg_latency,
            error_rate=error_rate,
            throughput=throughput,
            timestamp=time.time()
        )
        self._last_check[provider] = time.time()
    
    def check_degradation(self, provider: ModelProvider) -> Tuple[bool, Optional[str]]:
        """Check if model performance has degraded significantly."""
        if provider not in self._baselines or len(self._latencies[provider]) < 10:
            return False, None
            
        baseline = self._baselines[provider]
        recent_latencies = list(self._latencies[provider])[-10:]
        recent_errors = list(self._errors[provider])[-10:]
        
        if not recent_latencies:
            return False, None
            
        current_latency = sum(recent_latencies) / len(recent_latencies)
        current_error_rate = sum(recent_errors) / len(recent_errors)
        
        # Check latency degradation
        latency_increase = (current_latency - baseline.avg_latency) / baseline.avg_latency
        if latency_increase > self.degradation_threshold:
            return True, f"Latency increased by {latency_increase:.1%}"
            
        # Check error rate increase
        if baseline.error_rate > 0:
            error_increase = (current_error_rate - baseline.error_rate) / baseline.error_rate
            if error_increase > self.degradation_threshold:
                return True, f"Error rate increased by {error_increase:.1%}"
        elif current_error_rate > 0.05:  # 5% error rate threshold
            return True, f"Error rate spiked to {current_error_rate:.1%}"
            
        return False, None
    
    def get_performance_summary(self, provider: ModelProvider) -> Optional[Dict]:
        """Get current performance summary for a provider."""
        if provider not in self._baselines:
            return None
            
        baseline = self._baselines[provider]
        recent_latencies = list(self._latencies[provider])[-10:] if self._latencies[provider] else []
        recent_errors = list(self._errors[provider])[-10:] if self._errors[provider] else []
        
        current_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0
        current_error_rate = sum(recent_errors) / len(recent_errors) if recent_errors else 0
        
        return {
            "provider": provider.value,
            "baseline_latency": baseline.avg_latency,
            "current_latency": current_latency,
            "baseline_error_rate": baseline.error_rate,
            "current_error_rate": current_error_rate,
            "sample_count": len(self._latencies[provider])
        }
