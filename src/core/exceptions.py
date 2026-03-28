"""Custom exceptions for the LLM inference router."""

from typing import Optional, Dict, Any


class RouterError(Exception):
    """Base exception for router errors."""
    pass


class ModelUnavailableError(RouterError):
    """Raised when a model is unavailable."""
    
    def __init__(self, message: str, model_name: str, reason: Optional[str] = None):
        super().__init__(message)
        self.model_name = model_name
        self.reason = reason


class CircuitBreakerOpenError(RouterError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, message: str, model_name: str, failure_count: int):
        super().__init__(message)
        self.model_name = model_name
        self.failure_count = failure_count


class TimeoutError(RouterError):
    """Raised when a request times out."""
    
    def __init__(self, message: str, model_name: str, timeout: float, context: Dict[str, Any]):
        super().__init__(message)
        self.model_name = model_name
        self.timeout = timeout
        self.context = context


class ComplexityAnalysisError(RouterError):
    """Raised when complexity analysis fails."""
    pass


class LoadBalancerError(RouterError):
    """Raised when load balancing fails."""
    pass


class CacheError(RouterError):
    """Raised when cache operations fail."""
    pass
