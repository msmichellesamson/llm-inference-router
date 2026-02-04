import time
import logging
from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class ErrorType(Enum):
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERVER_ERROR = "server_error"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 30
    half_open_max_calls: int = 3
    exponential_backoff: bool = True
    max_backoff: int = 300

class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.last_failure_time: Optional[datetime] = None
        self.backoff_multiplier = 1
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if not self._should_attempt_reset():
                raise Exception(f"Circuit breaker {self.name} is OPEN")
            self._transition_to_half_open()
            
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            error_type = self._categorize_error(e)
            self._on_failure(error_type)
            raise
            
    def _categorize_error(self, error: Exception) -> ErrorType:
        error_str = str(error).lower()
        if "timeout" in error_str:
            return ErrorType.TIMEOUT
        elif "connection" in error_str:
            return ErrorType.CONNECTION
        elif "rate limit" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT
        elif "500" in error_str or "502" in error_str or "503" in error_str:
            return ErrorType.SERVER_ERROR
        return ErrorType.UNKNOWN
        
    def _on_success(self):
        self.failure_count = 0
        self.backoff_multiplier = 1
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info(f"Circuit breaker {self.name} transitioned to CLOSED")
            
    def _on_failure(self, error_type: ErrorType):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()
            
        logger.warning(f"Circuit breaker {self.name} failure #{self.failure_count}, type: {error_type.value}")
        
    def _transition_to_open(self):
        self.state = CircuitState.OPEN
        if self.config.exponential_backoff:
            self.backoff_multiplier = min(self.backoff_multiplier * 2, self.config.max_backoff // self.config.recovery_timeout)
        logger.error(f"Circuit breaker {self.name} transitioned to OPEN")
        
    def _transition_to_half_open(self):
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        logger.info(f"Circuit breaker {self.name} transitioned to HALF_OPEN")
        
    def _should_attempt_reset(self) -> bool:
        if not self.last_failure_time:
            return True
            
        timeout = self.config.recovery_timeout * self.backoff_multiplier
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= timeout
        
    @property
    def metrics(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "backoff_multiplier": self.backoff_multiplier,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }