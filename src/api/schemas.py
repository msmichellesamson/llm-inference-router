from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class ModelType(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"

class ComplexityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for LLM")
    model_preference: Optional[ModelType] = Field(None, description="Preferred model type")
    max_tokens: Optional[int] = Field(100, ge=1, le=4096, description="Maximum tokens")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    timeout_ms: Optional[int] = Field(30000, ge=1000, le=120000, description="Request timeout")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class InferenceResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    model_used: str = Field(..., description="Model that handled the request")
    complexity_score: float = Field(..., description="Analyzed complexity (0-1)")
    latency_ms: int = Field(..., description="Request latency in milliseconds")
    tokens_used: int = Field(..., description="Total tokens consumed")
    cached: bool = Field(..., description="Whether response was cached")
    cost_estimate: float = Field(..., description="Estimated cost in USD")

class HealthStatus(BaseModel):
    status: str = Field(..., description="Service status")
    models: Dict[str, bool] = Field(..., description="Model availability")
    cache_hit_rate: float = Field(..., description="Cache hit rate (0-1)")
    avg_latency_ms: float = Field(..., description="Average latency")
    uptime_seconds: int = Field(..., description="Service uptime")

class MetricsResponse(BaseModel):
    requests_total: int
    requests_per_model: Dict[str, int]
    avg_complexity: float
    cache_hit_rate: float
    error_rate: float
    p95_latency_ms: float