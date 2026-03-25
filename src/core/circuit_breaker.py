import asyncio
import time
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager
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
    max_backoff: int = 300
    backoff_multiplier: float = 2.0

class CircuitBreaker:
    def __init__(self, name: str, config: CircuitConfig = None):
        self.name = name
        self.config = config or CircuitConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.current_backoff = 1
        
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit {self.name} moving to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(f"Circuit {self.name} is OPEN")
                
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.last_failure_time:
            return True
            
        backoff_time = min(self.current_backoff, self.config.max_backoff)
        return time.time() - self.last_failure_time >= backoff_time
        
    def _on_success(self):
        """Reset circuit breaker on successful call"""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit {self.name} reset to CLOSED")
            
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.current_backoff = 1
        
    def _on_failure(self):
        """Handle failure and update circuit state"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.current_backoff = min(
                self.current_backoff * self.config.backoff_multiplier,
                self.config.max_backoff
            )
            logger.warning(
                f"Circuit {self.name} opened after {self.failure_count} failures. "
                f"Backoff: {self.current_backoff}s"
            )
            
class CircuitBreakerOpenError(Exception):
    pass