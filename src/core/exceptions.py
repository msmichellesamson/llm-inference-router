"""Custom exceptions for the LLM inference router."""

from typing import Optional, Dict, Any


class RouterError(Exception):
    """Base exception for router errors."""
    pass


class ModelUnavailableError(RouterError):
    """Raised when no models are available for routing."""
    pass


class ComplexityAnalysisError(RouterError):
    """Raised when complexity analysis fails."""
    pass


class CircuitBreakerError(RouterError):
    """Raised when circuit breaker is open."""
    pass


class TimeoutError(RouterError):
    """Raised when operation times out."""
    pass


class RateLimitError(RouterError):
    """Raised when rate limit is exceeded."""
    pass


class CacheError(RouterError):
    """Raised when cache operations fail."""
    pass


class ModelError(RouterError):
    """Raised when model inference fails."""
    
    def __init__(self, message: str, model_name: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.model_name = model_name
        self.retry_after = retry_after


class ValidationError(RouterError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str, value: Any):
        super().__init__(message)
        self.field = field
        self.value = value


class ConfigurationError(RouterError):
    """Raised when configuration is invalid."""
    pass


class HealthCheckError(RouterError):
    """Raised when health check fails."""
    pass


class LoadBalancerError(RouterError):
    """Raised when load balancer fails to select a model."""
    pass


class ShutdownError(RouterError):
    """Raised when graceful shutdown fails."""
    pass


class MetricsError(RouterError):
    """Raised when metrics collection fails."""
    pass