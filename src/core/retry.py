import asyncio
import random
import logging
from typing import Callable, Any, Optional
from functools import wraps

from .exceptions import RetryExhaustedException, CircuitBreakerOpenException

logger = logging.getLogger(__name__)

class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = min(self.base_delay * (self.backoff_multiplier ** attempt), self.max_delay)
        
        if self.jitter:
            # Add up to 25% jitter to prevent thundering herd
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount
            
        return delay

def retry_async(
    config: Optional[RetryConfig] = None,
    retryable_exceptions: tuple = (Exception,)
):
    """Async retry decorator with exponential backoff and jitter."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except CircuitBreakerOpenException:
                    # Don't retry if circuit breaker is open
                    raise
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"Retry exhausted for {func.__name__} after {config.max_attempts} attempts",
                            extra={"function": func.__name__, "attempts": config.max_attempts}
                        )
                        break
                    
                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.2f}s",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "delay": delay,
                            "error": str(e)
                        }
                    )
                    await asyncio.sleep(delay)
            
            raise RetryExhaustedException(
                f"Failed after {config.max_attempts} attempts",
                last_exception
            )
        
        return wrapper
    return decorator