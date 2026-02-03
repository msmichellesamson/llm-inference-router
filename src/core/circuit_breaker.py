from enum import Enum
from time import time
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, reset_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

    def can_execute(self) -> bool:
        """Check if request can be executed based on circuit state"""
        current_time = time()
        
        if self.state == CircuitState.OPEN:
            if current_time - (self.last_failure_time or 0) > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        
        return True

    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker CLOSED after successful request")

    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")

    def get_state(self) -> str:
        return self.state.value

class CircuitBreakerManager:
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, endpoint: str) -> CircuitBreaker:
        if endpoint not in self.breakers:
            self.breakers[endpoint] = CircuitBreaker()
        return self.breakers[endpoint]
    
    def is_available(self, endpoint: str) -> bool:
        return self.get_breaker(endpoint).can_execute()
    
    def record_success(self, endpoint: str):
        self.get_breaker(endpoint).record_success()
    
    def record_failure(self, endpoint: str):
        self.get_breaker(endpoint).record_failure()
    
    def get_status(self) -> Dict[str, str]:
        return {endpoint: breaker.get_state() for endpoint, breaker in self.breakers.items()}