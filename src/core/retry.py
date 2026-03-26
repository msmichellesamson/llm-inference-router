"""Retry mechanism with exponential backoff for model requests."""

import asyncio
import random
from typing import Any, Callable, Optional, TypeVar
from functools import wraps
import logging

from .exceptions import ModelUnavailableError, RateLimitError

T = TypeVar('T')

logger = logging.getLogger(__name__)

class RetryConfig:
    """Configuration for retry behavior."""
    
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

class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    RETRYABLE_EXCEPTIONS = (
        ModelUnavailableError,
        RateLimitError,
        ConnectionError,
        TimeoutError
    )
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff."""
        delay = min(
            self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1)),
            self.config.max_delay
        )
        
        if self.config.jitter:
            delay *= (0.5 + random.random() / 2)  # Add 0-50% jitter
        
        return delay
    
    async def retry_async(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        """Retry async function with exponential backoff."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                logger.debug(f"Attempt {attempt}/{self.config.max_attempts} for {func.__name__}")
                return await func(*args, **kwargs)
            
            except self.RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt} failed for {func.__name__}: {e}"
                )
                
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed")
                    break
            
            except Exception as e:
                logger.error(f"Non-retryable error in {func.__name__}: {e}")
                raise
        
        raise last_exception

def with_retry(config: RetryConfig = None):
    """Decorator to add retry behavior to async functions."""
    retry_handler = RetryHandler(config)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_handler.retry_async(func, *args, **kwargs)
        return wrapper
    
    return decorator