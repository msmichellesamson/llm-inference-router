"""Custom exceptions for LLM inference router."""

from enum import Enum
from typing import Optional


class ErrorCode(Enum):
    """Standard error codes for routing operations."""
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    TIMEOUT = "TIMEOUT"
    INVALID_REQUEST = "INVALID_REQUEST"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    RATE_LIMITED = "RATE_LIMITED"
    INFERENCE_ERROR = "INFERENCE_ERROR"


class RoutingError(Exception):
    """Base exception for routing errors."""
    
    def __init__(self, message: str, code: ErrorCode, model_id: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.model_id = model_id
        self.message = message

    def to_dict(self) -> dict:
        return {
            "error": self.message,
            "code": self.code.value,
            "model_id": self.model_id
        }


class ModelUnavailableError(RoutingError):
    """Raised when a model is unavailable."""
    
    def __init__(self, model_id: str, reason: str = "Model unavailable"):
        super().__init__(f"{reason}: {model_id}", ErrorCode.MODEL_UNAVAILABLE, model_id)


class CircuitOpenError(RoutingError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, model_id: str, retry_after: Optional[int] = None):
        message = f"Circuit open for model: {model_id}"
        if retry_after:
            message += f" (retry after {retry_after}s)"
        super().__init__(message, ErrorCode.CIRCUIT_OPEN, model_id)
        self.retry_after = retry_after


class QuotaExceededError(RoutingError):
    """Raised when model quota is exceeded."""
    
    def __init__(self, model_id: str, reset_time: Optional[int] = None):
        message = f"Quota exceeded for model: {model_id}"
        if reset_time:
            message += f" (resets at {reset_time})"
        super().__init__(message, ErrorCode.QUOTA_EXCEEDED, model_id)
        self.reset_time = reset_time
