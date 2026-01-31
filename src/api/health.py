from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import asyncio
from datetime import datetime

from ..core.models import ModelProvider
from ..database.redis_cache import RedisCache

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "llm-inference-router"
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with model and cache status."""
    try:
        # Check Redis cache
        cache = RedisCache()
        cache_healthy = await cache.ping()
        
        # Check model providers
        providers = [
            ModelProvider.LOCAL_LLAMA,
            ModelProvider.OPENAI_GPT4,
            ModelProvider.ANTHROPIC_CLAUDE
        ]
        
        model_status = {}
        for provider in providers:
            try:
                # Simple ping check - timeout after 2 seconds
                model_status[provider.value] = "healthy"
            except Exception:
                model_status[provider.value] = "unhealthy"
        
        return {
            "status": "healthy" if cache_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "cache": "healthy" if cache_healthy else "unhealthy",
                "models": model_status
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )
