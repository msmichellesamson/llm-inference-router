import time
import uuid
from typing import Dict, Optional
from contextvars import ContextVar
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

# Context variable for request tracing
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)

class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware to add request tracing and timing."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request_id_context.set(request_id)
        
        start_time = time.time()
        
        # Add to request state
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2)
                }
            )
            
            # Add tracing headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = str(round(duration * 1000, 2))
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "duration_ms": round(duration * 1000, 2)
                },
                exc_info=True
            )
            raise

def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_context.get()
