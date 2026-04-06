import asyncio
import random
from typing import Callable, Any, Optional
from functools import wraps
import logging
from .exceptions import RetryExhaustedException, CircuitBreakerOpenException

logger = logging.getLogger(__name__)

class ExponentialBackoff:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        
        if self.jitter:
            # Full jitter: random between 0 and calculated delay
            delay = random.uniform(0, delay)
        
        return delay
    
    async def __call__(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry and exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except CircuitBreakerOpenException:
                # Don't retry if circuit breaker is open
                raise
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) exceeded for {func.__name__}")
                    break
                
                delay = self.calculate_delay(attempt)
                logger.warning(
                    f"Retry attempt {attempt + 1}/{self.max_retries} for {func.__name__} "
                    f"after {delay:.2f}s. Error: {str(e)}"
                )
                
                await asyncio.sleep(delay)
        
        raise RetryExhaustedException(
            f"Failed after {self.max_retries} retries",
            original_exception=last_exception
        )

def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
):
    """Decorator to add exponential backoff retry to async functions."""
    def decorator(func: Callable) -> Callable:
        backoff = ExponentialBackoff(max_retries, base_delay, max_delay, jitter)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await backoff(func, *args, **kwargs)
        
        return wrapper
    return decorator
