from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryAction(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    SCALE_UP = "scale_up"
    ALERT_HUMAN = "alert_human"

@dataclass
class ErrorContext:
    error_type: str
    severity: ErrorSeverity
    component: str
    model_id: Optional[str]
    request_id: str
    timestamp: datetime
    details: Dict[str, Any]
    suggested_actions: list[RecoveryAction]
    retry_after_seconds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "severity": self.severity.value,
            "component": self.component,
            "model_id": self.model_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "suggested_actions": [action.value for action in self.suggested_actions],
            "retry_after_seconds": self.retry_after_seconds
        }

class ErrorContextBuilder:
    """Builder for creating structured error contexts with recovery guidance"""
    
    @staticmethod
    def timeout_error(
        component: str, 
        model_id: str, 
        request_id: str, 
        timeout_duration: float
    ) -> ErrorContext:
        return ErrorContext(
            error_type="timeout",
            severity=ErrorSeverity.HIGH,
            component=component,
            model_id=model_id,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            details={"timeout_duration_seconds": timeout_duration},
            suggested_actions=[RecoveryAction.FALLBACK, RecoveryAction.SCALE_UP],
            retry_after_seconds=30
        )
    
    @staticmethod
    def rate_limit_error(
        component: str, 
        model_id: str, 
        request_id: str, 
        retry_after: int
    ) -> ErrorContext:
        return ErrorContext(
            error_type="rate_limit",
            severity=ErrorSeverity.MEDIUM,
            component=component,
            model_id=model_id,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            details={"rate_limit_reset_seconds": retry_after},
            suggested_actions=[RecoveryAction.RETRY, RecoveryAction.FALLBACK],
            retry_after_seconds=retry_after
        )
    
    @staticmethod
    def model_unavailable(
        component: str, 
        model_id: str, 
        request_id: str, 
        health_status: str
    ) -> ErrorContext:
        return ErrorContext(
            error_type="model_unavailable",
            severity=ErrorSeverity.CRITICAL,
            component=component,
            model_id=model_id,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            details={"health_status": health_status},
            suggested_actions=[RecoveryAction.FALLBACK, RecoveryAction.CIRCUIT_BREAK],
            retry_after_seconds=300
        )