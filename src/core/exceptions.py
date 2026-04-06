class LLMRouterException(Exception):
    """Base exception for LLM router."""
    pass

class ModelUnavailableException(LLMRouterException):
    """Raised when a model is unavailable."""
    pass

class ComplexityAnalysisException(LLMRouterException):
    """Raised when complexity analysis fails."""
    pass

class CircuitBreakerOpenException(LLMRouterException):
    """Raised when circuit breaker is open."""
    pass

class RetryExhaustedException(LLMRouterException):
    """Raised when retry attempts are exhausted."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception

class TimeoutException(LLMRouterException):
    """Raised when operation times out."""
    pass

class CacheException(LLMRouterException):
    """Raised when cache operations fail."""
    pass

class LoadBalancerException(LLMRouterException):
    """Raised when load balancer fails to select instance."""
    pass
