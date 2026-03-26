import asyncio
import random
from typing import Callable, Any, Optional
from functools import wraps
from .exceptions import ModelTimeoutError, ModelUnavailableError
from .metrics import MetricsCollector


class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter


def with_retry(config: RetryConfig, metrics: MetricsCollector):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        metrics.increment('retry_success_total', {'attempt': str(attempt + 1)})
                    return result
                    
                except (ModelTimeoutError, ModelUnavailableError, ConnectionError) as e:
                    last_exception = e
                    metrics.increment('retry_attempt_total', {
                        'exception': type(e).__name__,
                        'attempt': str(attempt + 1)
                    })
                    
                    if attempt < config.max_attempts - 1:
                        delay = min(
                            config.base_delay * (config.backoff_factor ** attempt),
                            config.max_delay
                        )
                        
                        if config.jitter:
                            delay *= (0.5 + random.random() * 0.5)
                        
                        await asyncio.sleep(delay)
                    
                except Exception as e:
                    # Non-retryable errors fail immediately
                    metrics.increment('retry_non_retryable_total', {
                        'exception': type(e).__name__
                    })
                    raise
            
            metrics.increment('retry_exhausted_total')
            raise last_exception
            
        return wrapper
    return decorator


# Global retry configurations
DEFAULT_RETRY = RetryConfig(max_attempts=3, base_delay=1.0)
AGGRESSIVE_RETRY = RetryConfig(max_attempts=5, base_delay=0.5, backoff_factor=1.5)
CONSERVATIVE_RETRY = RetryConfig(max_attempts=2, base_delay=2.0, backoff_factor=3.0)