from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Optional
import logging

from ..core.models import ModelType
from ..core.exceptions import ModelNotFoundError
from .health import get_redis_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/temperature", tags=["temperature"])

class TemperatureRequest(BaseModel):
    model_type: ModelType
    temperature: float = Field(ge=0.0, le=2.0, description="Temperature between 0.0 and 2.0")
    duration_minutes: Optional[int] = Field(default=30, ge=1, le=1440, description="Duration in minutes")

class TemperatureResponse(BaseModel):
    model_type: ModelType
    old_temperature: float
    new_temperature: float
    expires_in_minutes: int

@router.post("/adjust", response_model=TemperatureResponse)
async def adjust_temperature(
    request: TemperatureRequest,
    redis_client=Depends(get_redis_client)
):
    """Temporarily adjust model temperature for experimentation."""
    try:
        # Get current temperature (default 0.7)
        temp_key = f"temp:{request.model_type.value}"
        current_temp = await redis_client.get(temp_key)
        old_temperature = float(current_temp) if current_temp else 0.7
        
        # Set new temperature with TTL
        ttl_seconds = request.duration_minutes * 60
        await redis_client.setex(temp_key, ttl_seconds, str(request.temperature))
        
        logger.info(
            f"Temperature adjusted for {request.model_type.value}: "
            f"{old_temperature} -> {request.temperature} for {request.duration_minutes}m"
        )
        
        return TemperatureResponse(
            model_type=request.model_type,
            old_temperature=old_temperature,
            new_temperature=request.temperature,
            expires_in_minutes=request.duration_minutes
        )
    except Exception as e:
        logger.error(f"Failed to adjust temperature: {e}")
        raise HTTPException(status_code=500, detail="Temperature adjustment failed")

@router.get("/current/{model_type}", response_model=Dict[str, float])
async def get_current_temperature(
    model_type: ModelType,
    redis_client=Depends(get_redis_client)
):
    """Get current temperature setting for a model."""
    try:
        temp_key = f"temp:{model_type.value}"
        current_temp = await redis_client.get(temp_key)
        temperature = float(current_temp) if current_temp else 0.7
        
        # Get TTL if exists
        ttl = await redis_client.ttl(temp_key)
        
        return {
            "temperature": temperature,
            "ttl_seconds": ttl if ttl > 0 else None
        }
    except Exception as e:
        logger.error(f"Failed to get temperature: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve temperature")

@router.delete("/reset/{model_type}")
async def reset_temperature(
    model_type: ModelType,
    redis_client=Depends(get_redis_client)
):
    """Reset model temperature to default (0.7)."""
    try:
        temp_key = f"temp:{model_type.value}"
        await redis_client.delete(temp_key)
        
        logger.info(f"Temperature reset to default for {model_type.value}")
        return {"message": f"Temperature reset to default for {model_type.value}"}
    except Exception as e:
        logger.error(f"Failed to reset temperature: {e}")
        raise HTTPException(status_code=500, detail="Temperature reset failed")