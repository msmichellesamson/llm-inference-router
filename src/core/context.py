import contextvars
import uuid
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RequestContext:
    """Request context for tracking across services."""
    correlation_id: str
    user_id: Optional[str] = None
    model_preference: Optional[str] = None
    priority: str = "normal"
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

# Context variable for async request tracking
_request_context: contextvars.ContextVar[Optional[RequestContext]] = \
    contextvars.ContextVar('request_context', default=None)

def get_context() -> Optional[RequestContext]:
    """Get current request context."""
    return _request_context.get()

def set_context(context: RequestContext) -> None:
    """Set request context."""
    _request_context.set(context)

def create_context(user_id: Optional[str] = None, 
                  model_preference: Optional[str] = None,
                  priority: str = "normal",
                  **metadata) -> RequestContext:
    """Create new request context with correlation ID."""
    context = RequestContext(
        correlation_id=str(uuid.uuid4()),
        user_id=user_id,
        model_preference=model_preference,
        priority=priority,
        metadata=metadata
    )
    set_context(context)
    return context

def get_correlation_id() -> Optional[str]:
    """Get correlation ID from current context."""
    context = get_context()
    return context.correlation_id if context else None

def clear_context() -> None:
    """Clear current request context."""
    _request_context.set(None)
