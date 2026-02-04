import asyncio
import random
import time
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3
    max_backoff: int = 300
    base_delay: float = 1.0

class CircuitBreaker:
    def __init__(self, name: str, config: Optional[CircuitConfig] = None):
        self.name = name
        self.config = config or CircuitConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.next_attempt_time = 0
        
    def _calculate_backoff_delay(self) -> float:
        """Calculate exponential backoff with jitter"""
        if self.failure_count == 0:
            return 0
            
        # Exponential backoff: base_delay * 2^(failure_count-1)
        delay = self.config.base_delay * (2 ** (self.failure_count - 1))
        delay = min(delay, self.config.max_backoff)
        
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0, delay + jitter)
        
    def can_attempt(self) -> bool:
        """Check if request can be attempted based on circuit state"""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
            
        if self.state == CircuitState.OPEN:
            if current_time >= self.next_attempt_time:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                return True
            return False
            
        # HALF_OPEN state
        return True
        
    def record_success(self):
        """Record successful request"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} recovered to CLOSED")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
            
    def record_failure(self):
        """Record failed request with exponential backoff"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                backoff_delay = self._calculate_backoff_delay()
                self.next_attempt_time = time.time() + backoff_delay
                logger.warning(
                    f"Circuit breaker {self.name} opened. Next attempt in {backoff_delay:.2f}s"
                )
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            backoff_delay = self._calculate_backoff_delay()
            self.next_attempt_time = time.time() + backoff_delay
            logger.warning(f"Circuit breaker {self.name} failed in HALF_OPEN, reopened")
            
    def get_stats(self) -> Dict:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "next_attempt_time": self.next_attempt_time if self.state == CircuitState.OPEN else None
        }