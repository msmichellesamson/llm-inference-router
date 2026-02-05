from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class ErrorCategory(Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    NETWORK = "network"
    AUTHENTICATION = "auth"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    timeout_seconds: int = 60
    half_open_max_calls: int = 3
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 300.0
    backoff_multiplier: float = 2.0

class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self.consecutive_failures = 0
        self.error_counts: Dict[ErrorCategory, int] = {cat: 0 for cat in ErrorCategory}
        
    def _calculate_backoff(self) -> float:
        """Calculate exponential backoff with jitter"""
        if self.consecutive_failures == 0:
            return self.config.base_backoff_seconds
            
        backoff = min(
            self.config.base_backoff_seconds * (self.config.backoff_multiplier ** (self.consecutive_failures - 1)),
            self.config.max_backoff_seconds
        )
        # Add jitter (±20%)
        import random
        jitter = backoff * 0.2 * (random.random() - 0.5)
        return max(0.1, backoff + jitter)
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error for better circuit breaker decisions"""
        error_str = str(error).lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return ErrorCategory.TIMEOUT
        elif "rate limit" in error_str or "429" in error_str:
            return ErrorCategory.RATE_LIMIT  
        elif "auth" in error_str or "401" in error_str or "403" in error_str:
            return ErrorCategory.AUTHENTICATION
        elif "connection" in error_str or "network" in error_str:
            return ErrorCategory.NETWORK
        else:
            return ErrorCategory.SERVER_ERROR
    
    def call(self, func, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise Exception(f"Circuit breaker {self.name} HALF_OPEN call limit exceeded")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        if not self.last_failure_time:
            return True
        
        backoff_time = self._calculate_backoff()
        return datetime.now() >= self.last_failure_time + timedelta(seconds=backoff_time)
    
    def _on_success(self):
        self.failure_count = 0
        self.consecutive_failures = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info(f"Circuit breaker {self.name} reset to CLOSED")
        
        if self.state != CircuitState.CLOSED:
            self.half_open_calls += 1
    
    def _on_failure(self, error: Exception):
        error_category = self._categorize_error(error)
        self.error_counts[error_category] += 1
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now()
        
        # Don't trip on auth errors - they're not service health issues
        if error_category == ErrorCategory.AUTHENTICATION:
            return
            
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.name} OPENED after {self.failure_count} failures")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "error_counts": {cat.value: count for cat, count in self.error_counts.items()},
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }