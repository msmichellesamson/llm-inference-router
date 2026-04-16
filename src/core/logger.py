"""Structured logging with correlation IDs for request tracing."""

import logging
import uuid
from contextvars import ContextVar
from typing import Dict, Any, Optional
import json
import time

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': time.time(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation ID if present
        corr_id = correlation_id.get()
        if corr_id:
            log_data['correlation_id'] = corr_id
        
        # Add extra fields from log record
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
            
        return json.dumps(log_data)

def get_logger(name: str) -> logging.Logger:
    """Get configured logger with structured formatting."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

def set_correlation_id(request_id: Optional[str] = None) -> str:
    """Set correlation ID for current context."""
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    
    correlation_id.set(request_id)
    return request_id

def log_with_context(logger: logging.Logger, level: int, msg: str, **kwargs) -> None:
    """Log with additional context fields."""
    extra_fields = kwargs
    logger.log(level, msg, extra={'extra_fields': extra_fields})

def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id.get()
