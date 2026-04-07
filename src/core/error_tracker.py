"""Error tracking and categorization for detailed observability."""

import time
from collections import defaultdict, Counter
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from threading import Lock

class ErrorCategory(Enum):
    """Error categories for classification."""
    MODEL_TIMEOUT = "model_timeout"
    MODEL_OVERLOAD = "model_overload"
    NETWORK_ERROR = "network_error"
    INVALID_INPUT = "invalid_input"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    CIRCUIT_BREAKER = "circuit_breaker"
    UNKNOWN = "unknown"

@dataclass
class ErrorMetrics:
    """Metrics for a specific error category."""
    count: int = 0
    last_occurrence: Optional[float] = None
    avg_recovery_time: float = 0.0
    total_recovery_time: float = 0.0
    recovery_samples: int = 0

    def add_occurrence(self, recovery_time: Optional[float] = None):
        """Record an error occurrence."""
        self.count += 1
        self.last_occurrence = time.time()
        
        if recovery_time is not None:
            self.total_recovery_time += recovery_time
            self.recovery_samples += 1
            self.avg_recovery_time = self.total_recovery_time / self.recovery_samples

class ErrorTracker:
    """Tracks and categorizes errors for detailed metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, Dict[ErrorCategory, ErrorMetrics]] = defaultdict(
            lambda: {cat: ErrorMetrics() for cat in ErrorCategory}
        )
        self._lock = Lock()
        
    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on type and message."""
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()
        
        if "timeout" in error_msg or "timeout" in error_type:
            return ErrorCategory.MODEL_TIMEOUT
        elif "overload" in error_msg or "busy" in error_msg:
            return ErrorCategory.MODEL_OVERLOAD
        elif "connection" in error_msg or "network" in error_msg:
            return ErrorCategory.NETWORK_ERROR
        elif "invalid" in error_msg or "validation" in error_msg:
            return ErrorCategory.INVALID_INPUT
        elif "rate limit" in error_msg or "quota" in error_msg:
            return ErrorCategory.RATE_LIMIT
        elif "auth" in error_msg or "permission" in error_msg:
            return ErrorCategory.AUTHENTICATION
        elif "circuit" in error_msg:
            return ErrorCategory.CIRCUIT_BREAKER
        else:
            return ErrorCategory.UNKNOWN
    
    def track_error(self, model_name: str, error: Exception, 
                   recovery_time: Optional[float] = None):
        """Track an error occurrence."""
        category = self.categorize_error(error)
        
        with self._lock:
            self._metrics[model_name][category].add_occurrence(recovery_time)
    
    def get_error_stats(self, model_name: str) -> Dict[str, Dict]:
        """Get error statistics for a model."""
        with self._lock:
            model_metrics = self._metrics.get(model_name, {})
            
            stats = {}
            for category, metrics in model_metrics.items():
                if metrics.count > 0:
                    stats[category.value] = {
                        "count": metrics.count,
                        "last_occurrence": metrics.last_occurrence,
                        "avg_recovery_time": metrics.avg_recovery_time
                    }
            
            return stats
    
    def get_top_errors(self, limit: int = 5) -> List[Dict]:
        """Get top errors across all models."""
        error_counts = Counter()
        
        with self._lock:
            for model_name, categories in self._metrics.items():
                for category, metrics in categories.items():
                    if metrics.count > 0:
                        key = f"{model_name}:{category.value}"
                        error_counts[key] = metrics.count
        
        return [
            {"error": error, "count": count}
            for error, count in error_counts.most_common(limit)
        ]