class LLMRouterException(Exception):
    """Base exception for LLM Router"""
    pass

class ModelUnavailable(LLMRouterException):
    """Raised when a model is unavailable"""
    pass

class ComplexityAnalysisError(LLMRouterException):
    """Raised when complexity analysis fails"""
    pass

class CircuitBreakerOpen(LLMRouterException):
    """Raised when circuit breaker is open"""
    pass

class TimeoutError(LLMRouterException):
    """Raised when request times out"""
    pass

class RetryExhausted(LLMRouterException):
    """Raised when all retries are exhausted"""
    pass

class RateLimitExceeded(LLMRouterException):
    """Raised when rate limit is exceeded"""
    pass

class ValidationError(LLMRouterException):
    """Raised when input validation fails"""
    pass

class DatabaseError(LLMRouterException):
    """Raised when database operations fail"""
    pass

class ModelDriftDetected(LLMRouterException):
    """Raised when model drift is detected"""
    pass
