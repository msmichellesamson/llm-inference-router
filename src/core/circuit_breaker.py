import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class ErrorType(Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    AUTHENTICATION = "auth_error"
    NETWORK = "network_error"

@dataclass
class CircuitConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    timeout_threshold: float = 30.0
    max_backoff: int = 300

class CircuitBreaker:
    def __init__(self, name: str, config: CircuitConfig = None):
        self.name = name
        self.config = config or CircuitConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self.backoff_multiplier = 1
        self.error_counts: Dict[ErrorType, int] = {}
        
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for better handling"""
        error_msg = str(error).lower()
        
        if "timeout" in error_msg or "timed out" in error_msg:
            return ErrorType.TIMEOUT
        elif "rate limit" in error_msg or "429" in error_msg:
            return ErrorType.RATE_LIMIT
        elif "authentication" in error_msg or "unauthorized" in error_msg:
            return ErrorType.AUTHENTICATION
        elif "connection" in error_msg or "network" in error_msg:
            return ErrorType.NETWORK
        else:
            return ErrorType.SERVER_ERROR
    
    def _get_backoff_delay(self) -> int:
        """Calculate exponential backoff delay"""
        delay = min(
            self.config.recovery_timeout * self.backoff_multiplier,
            self.config.max_backoff
        )
        return delay
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset"""
        if self.last_failure_time is None:
            return False
            
        backoff_delay = self._get_backoff_delay()
        return datetime.now() - self.last_failure_time >= timedelta(seconds=backoff_delay)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"Circuit breaker {self.name}: Transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise Exception(f"Circuit breaker {self.name} half-open call limit exceeded")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.backoff_multiplier = 1
            logger.info(f"Circuit breaker {self.name}: Reset to CLOSED")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, error: Exception):
        """Handle failed call with error classification"""
        error_type = self._classify_error(error)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.backoff_multiplier = min(self.backoff_multiplier * 2, 8)
            logger.warning(f"Circuit breaker {self.name}: Back to OPEN from half-open")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.backoff_multiplier = 2
            logger.warning(f"Circuit breaker {self.name}: Opened due to {self.failure_count} failures")
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "error_breakdown": dict(self.error_counts),
            "backoff_multiplier": self.backoff_multiplier,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }